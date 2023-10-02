import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from dataclasses_json import Undefined, dataclass_json

from ocpapi.client import (
    Adsorbates,
    AdsorbateSlabConfigs,
    AdsorbateSlabRelaxationResult,
    AdsorbateSlabRelaxationsResults,
    AdsorbateSlabRelaxationsSystem,
    Atoms,
    Bulk,
    Bulks,
    Client,
    Model,
    Slab,
    Slabs,
    Status,
)

from .context import set_context_var
from .retry import NO_LIMIT, RateLimitLogging, retry_api_calls

log = logging.getLogger(__name__)

# Context instance that stores information about the adsorbate and bulk
# material as a tuple in that order
_CTX_AD_BULK: ContextVar[Tuple[str, Bulk]] = ContextVar(f"{__name__}:ad_bulk")

# Context intance that stores information about a slab
_CTX_SLAB: ContextVar[Slab] = ContextVar(f"{__name__}:slab")


def _setup_log_record_factory() -> None:
    """
    Adds a log record factory that stores information about the currently
    running job on a log message.
    """
    old_factory: Callable[..., logging.LogRecord] = logging.getLogRecordFactory()

    def new_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        # Save information about the bulk and absorbate if set
        parts: List[str] = []
        if (ad_bulk := _CTX_AD_BULK.get(None)) is not None:
            parts.append(f"[{ad_bulk[0]}/{ad_bulk[1]}]")

        # Save information about the slab if set
        if (slab := _CTX_SLAB.get(None)) is not None:
            m = slab.metadata
            top = "t" if m.top else "b"
            millers = f"({m.millers[0]},{m.millers[1]},{m.millers[2]})"
            parts.append(f"[{millers}/{round(m.shift, 3):.3f},{top}]")

        # Prepend context to the current message
        record = old_factory(*args, **kwargs)
        parts.append(record.msg)
        record.msg = " ".join(parts)
        return record

    logging.setLogRecordFactory(new_factory)


_setup_log_record_factory()


DEFAULT_CLIENT: Client = Client()


class UnsupportedBulkException(Exception):
    """
    Exception raised when a bulk material is not supported in the API.
    """

    def __init__(self, bulk: str) -> None:
        """
        Args:
            bulk: The bulk structure that was requested.
        """
        super().__init__(f"Bulk {bulk} is not supported")


class UnsupportedAdsorbateException(Exception):
    """
    Exception raised when an adsorbate is not supported in the API.
    """

    def __init__(self, adsorbate: str) -> None:
        """
        Args:
            adsorbate: The adsorbate that was requested.
        """
        super().__init__(f"Adsorbate {adsorbate} is not supported")


class Lifetime(Enum):
    """
    Represents different lifetimes when running relaxations.

    Attributes:
        SAVE: The relaxation will be available on API servers indefinitely.
            It will not be possible to delete the relaxation in the future.
        MARK_EPHEMERAL: The relaxation will be saved on API servers, but
            can be deleted at any time in the future.
        DELETE: The relaxation will be deleted from API servers as soon as
            the results have been fetched.
    """

    SAVE = auto()
    MARK_EPHEMERAL = auto()
    DELETE = auto()


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass(kw_only=True)
class AdsorbateSlabRelaxation(AdsorbateSlabRelaxationResult):
    """
    Extension of AdsorbateSlabRelaxationResult that includes information
    about the initial structure and other inputs to the relaxation.

    Attributes:
        adsorbate: The SMILES string of the adsorbate.
        adsorbate_config: The configuration of the adsorbate before relaxation.
        bulk: The bulk material that was being modeled.
        slab: The slab on which the adsorbate was placed.
        model: The type of the model that was run.
    """

    adsorbate: str
    adsorbate_config: Atoms
    bulk: Bulk
    slab: Slab
    model: Model


@retry_api_calls(max_attempts=3)
async def _get_bulk_if_supported(client: Client, bulk: str) -> Bulk:
    """
    Returns the object from the input bulk if it is supported in the API.

    Args:
        client: The client to use when making requests to the API.
        bulk: The bulk to fetch.

    Raises:
        UnsupportedBulkException if the requested bulk is not supported.

    Returns:
        Bulk instance for the input type.
    """
    bulks: Bulks = await client.get_bulks()
    for b in bulks.bulks_supported:
        if b.src_id == bulk:
            return b
    raise UnsupportedBulkException(bulk)


@retry_api_calls(max_attempts=3)
async def _ensure_adsorbate_supported(client: Client, adsorbate: str) -> None:
    """
    Checks that the input adsorbate is supported in the API.

    Args:
        client: The client to use when making requests to the API.
        adsorbate: The adsorbate to check.

    Raises:
        UnsupportedAdsorbateException if the adsorbate is not supported.
    """
    adsorbates: Adsorbates = await client.get_adsorbates()
    if adsorbate not in adsorbates.adsorbates_supported:
        raise UnsupportedAdsorbateException(adsorbate)


@retry_api_calls(max_attempts=3)
async def _get_slabs(
    client: Client,
    bulk: Bulk,
    slab_filter: Optional[Callable[[Slab], bool]],
) -> List[Slab]:
    """
    Enumerates surfaces for the input bulk material.

    Args:
        client: The client to use when making requests to the API.
        bulk: The bulk material from which slabs will be generated.
        slab_filter: If not None, a function that filters which generated
            slabs will be considered when placing adsorbates.
    """
    slabs: Slabs = await client.get_slabs(bulk)
    slabs_list: List[slabs] = slabs.slabs
    if slab_filter is not None:
        slabs_list = [s for s in slabs_list if slab_filter(s)]
    return slabs_list


@retry_api_calls(max_attempts=3)
async def _get_absorbate_configs_on_slab(
    client: Client,
    adsorbate: str,
    slab: Slab,
) -> List[Atoms]:
    """
    Generate initial guesses at adsorbate binding sites on the input slab.

    Args:
        client: The client to use when making API calls.
        adsorbate: The SMILES string of the adsorbate to place.
        slab: The slab on which the adsorbate should be placed.

    Returns:
        List of Atoms objects, each with the positions of the adsorbate atoms
        on one of the candidate binding sites.
    """
    configs: AdsorbateSlabConfigs = await client.get_adsorbate_slab_configs(
        adsorbate=adsorbate,
        slab=slab,
    )
    return configs.adsorbate_configs


# The API behind Client.submit_adsorbate_slab_relaxations() is rate limited
# and this decorator will handle retrying when that rate limit is breached.
# Retry forever since we can't know how many jobs are being submitted along
# with this one (rate limits are enforced on the API server and not limited
# to a specific instance of this module).
@retry_api_calls(
    max_attempts=NO_LIMIT,
    rate_limit_logging=RateLimitLogging(
        logger=log,
        action="submit relaxations",
    ),
)
async def _submit_relaxations(
    client: Client,
    adsorbate: str,
    adsorbate_configs: List[Atoms],
    bulk: Bulk,
    slab: Slab,
    model: Model,
    ephemeral: bool,
) -> str:
    """
    Start relaxations for each of the input adsorbate configurations on the
    input slab.

    Args:
        client: The client to use when making API calls.
        adsorbate: The SMILES string of the adsorbate to place.
        adsorbate_configs: Positions of the adsorbate on the slab. Each
            will be relaxed independently.
        bulk: The bulk material from which the slab was generated.
        slab: The slab that should be searched for adsorbate binding sites.
        model: The model to use when evaluating forces and energies.
        ephemeral: Whether the relaxations should be marked as ephemeral.

    Returns:
        The system ID of the relaxation run, which can be used to fetch results
        as they become available.
    """
    system: AdsorbateSlabRelaxationsSystem = (
        await client.submit_adsorbate_slab_relaxations(
            adsorbate=adsorbate,
            adsorbate_configs=adsorbate_configs,
            bulk=bulk,
            slab=slab,
            model=model,
            ephemeral=ephemeral,
        )
    )
    return system.system_id


async def _submit_relaxations_with_logging(
    client: Client,
    adsorbate: str,
    adsorbate_configs: List[Atoms],
    bulk: Bulk,
    slab: Slab,
    model: Model,
    ephemeral: bool,
) -> str:
    """
    Wrapper around _submit_relaxations that adds periodic logging in case
    calls to submit relaxations are being rate limited.
    """

    # Function that will log periodically while attempts to submit relaxations
    # are being retried
    async def log_waiting() -> None:
        while True:
            await asyncio.sleep(30)
            log.info(
                "Still waiting for relaxations to be accepted, possibly "
                "because calls are being rate limited"
            )

    # Run until relaxations are accepted
    submit_task = asyncio.create_task(
        _submit_relaxations(
            client=client,
            adsorbate=adsorbate,
            adsorbate_configs=adsorbate_configs,
            bulk=bulk,
            slab=slab,
            model=model,
            ephemeral=ephemeral,
        )
    )
    _, pending = await asyncio.wait(
        [
            asyncio.create_task(log_waiting()),
            submit_task,
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel pending tasks (this should just be the task to log that waiting)
    for task in pending:
        with suppress(asyncio.CancelledError):
            task.cancel()
            await task

    return submit_task.result()


@retry_api_calls(max_attempts=3)
async def get_adsorbate_slab_relaxation_results(
    system_id: str,
    config_ids: Optional[List[int]] = None,
    fields: Optional[List[str]] = None,
    client: Client = DEFAULT_CLIENT,
) -> List[AdsorbateSlabRelaxationResult]:
    """
    Wrapper around Client.get_adsorbate_slab_relaxations_results() that
    handles retries, including re-fetching individual configurations that
    are initially omitted.

    Args:
        client: The client to use when making API calls.
        system_id: The system ID of the relaxations.
        config_ids: If defined and not empty, a subset of configurations
            to fetch. Otherwise all configurations are returned.
        fields: If defined and not empty, a subset of fields in each
            configuration to fetch. Otherwise all fields are returned.

    Returns:
        List of relaxation results, one for each adsorbate configuration in
        the system.
    """
    results: AdsorbateSlabRelaxationsResults = (
        await client.get_adsorbate_slab_relaxations_results(
            system_id=system_id,
            config_ids=config_ids,
            fields=fields,
        )
    )

    # Save a copy of all results that were fetched
    fetched: List[AdsorbateSlabRelaxationResult] = list(results.configs)

    # If any results were omitted, fetch them before returning
    if results.omitted_config_ids:
        fetched.extend(
            await get_adsorbate_slab_relaxation_results(
                client=client,
                system_id=system_id,
                config_ids=results.omitted_config_ids,
                fields=fields,
            )
        )

    return fetched


async def wait_for_adsorbate_slab_relaxations(
    system_id: str,
    check_immediately: bool = False,
    slow_interval_sec: float = 30,
    fast_interval_sec: float = 10,
    client: Client = DEFAULT_CLIENT,
) -> Dict[int, Status]:
    """
    Blocks until all relaxations in the input system have finished, whether
    successfully or not.

    Relaxations are queued in the API, waiting until machines are ready to
    run them. Once started, they can take 1-2 minutes to finish. This method
    initially sleeps "slow_interval_sec" seconds between each check for any
    relaxations having finished. Once at least one result is ready, subsequent
    sleeps are for "fast_interval_sec" seconds.

    Args:
        client: The client to use when making API calls.
        system_id: The ID of the system for which relaxations are running.
        check_immediately: If False (default), sleep before the first check
            for relaxations having finished. If True, check whether relaxations
            have finished immediately on entering this function.
        slow_interval_sec: The number of seconds to wait between each check
            while all are still running.
        fast_interval_sec: The number of seconds to wait between each check
            when at least one relaxation has finished in the system.

    Returns:
        Map of config IDs in the system to their terminal status.
    """

    # First wait if needed
    wait_for_sec: float = slow_interval_sec
    if not check_immediately:
        await asyncio.sleep(wait_for_sec)

    # Run until all results are available
    while True:
        # Get the current results. Only fetch the energy; this hits an index
        # that will return results more quickly.
        results: List[
            AdsorbateSlabRelaxationResult
        ] = await get_adsorbate_slab_relaxation_results(
            client=client,
            system_id=system_id,
            fields=["energy"],
        )

        # Return if all of the relaxations have finished
        unique_statuses: Set[Status] = set(r.status for r in results)
        if Status.NOT_AVAILABLE not in unique_statuses:
            log.info("All relaxations have finished")
            return {r.config_id: r.status for r in results}

        # Shorten the wait time if any relaxations have finished. not_available
        # means that a relaxation is still running, so the presence of any
        # other statuses means that some have finished.
        if unique_statuses - {Status.NOT_AVAILABLE}:
            wait_for_sec = fast_interval_sec

        # Wait until the next scheduled check
        num_finished: int = len(
            [r for r in results if r.status != Status.NOT_AVAILABLE]
        )
        log.info(f"{num_finished} of {len(results)} relaxations have finished")
        await asyncio.sleep(wait_for_sec)


@retry_api_calls(max_attempts=3)
async def _delete_system(client: Client, system_id: str) -> None:
    """
    Deletes the input system, with retries on failed attempts.

    Args:
        client: The client to use when making API calls.
        system_id: The ID of the system to delete.
    """
    await client.delete_adsorbate_slab_relaxations(system_id)


@asynccontextmanager
async def _ensure_system_deleted(
    client: Client,
    system_id: str,
) -> AsyncGenerator[None, None]:
    """
    Immediately yields control to the caller. When control returns to this
    function, try to delete the system with the input id.

    Args:
        client: The client to use when making API calls.
        system_id: The ID of the system to delete.
    """
    try:
        yield
    finally:
        log.info(f"Ensuring system with id {system_id} is deleted")
        await _delete_system(client=client, system_id=system_id)


async def _find_binding_sites_on_slab(
    client: Client,
    adsorbate: str,
    bulk: Bulk,
    slab: Slab,
    model: Model,
    lifetime: Lifetime = Lifetime.SAVE,
) -> List[AdsorbateSlabRelaxation]:
    """
    Search for adsorbate binding sites on the input slab.

    Args:
        client: The client to use when making API calls.
        adsorbate: The SMILES string of the adsorbate to place.
        bulk: The bulk material from which the slab was generated.
        slab: The slab that should be searched for adsorbate binding sites.
        model: The model to use when evaluating forces and energies.
        lifetime: Whether relaxations should be saved on the server, be marked
            as ephemeral (allowing them to deleted in the future), or deleted
            immediately.

    Returns:
        Details of each adsorbate placement, including its relaxed position.
    """
    async with AsyncExitStack() as es:
        es.enter_context(set_context_var(_CTX_SLAB, slab))

        # Enumerate candidate binding sites
        log.info(
            "Generating adsorbate placements on "
            f"{'top' if slab.metadata.top else 'bottom'} "
            f"{slab.metadata.millers} surface, shifted by "
            f"{round(slab.metadata.shift, 3)}"
        )
        adsorbate_configs: List[Atoms] = await _get_absorbate_configs_on_slab(
            client=client,
            adsorbate=adsorbate,
            slab=slab,
        )

        # Start relaxations for all of the adsorbate placements
        log.info(
            f"Submitting relaxations for {len(adsorbate_configs)} "
            "adsorbate placements"
        )
        system_id: str = await _submit_relaxations_with_logging(
            client=client,
            adsorbate=adsorbate,
            adsorbate_configs=adsorbate_configs,
            bulk=bulk,
            slab=slab,
            model=model,
            ephemeral=lifetime in {Lifetime.MARK_EPHEMERAL, Lifetime.DELETE},
        )
        log.info(f"Relaxations running with system id {system_id}")

        # If requested, ensure the system is deleted once results have been
        # fetched
        if lifetime == Lifetime.DELETE:
            await es.enter_async_context(
                _ensure_system_deleted(client=client, system_id=system_id)
            )

        # Wait for all relaxations to finish
        await wait_for_adsorbate_slab_relaxations(
            client=client,
            system_id=system_id,
        )

        # Fetch the final results
        results: List[
            AdsorbateSlabRelaxationResult
        ] = await get_adsorbate_slab_relaxation_results(
            client=client,
            system_id=system_id,
        )
        return [
            AdsorbateSlabRelaxation(
                adsorbate=adsorbate,
                adsorbate_config=initial_config,
                bulk=bulk,
                slab=slab,
                model=model,
                **vars(result),
            )
            for initial_config, result in zip(adsorbate_configs, results)
        ]


async def find_adsorbate_binding_sites(
    adsorbate: str,
    bulk: str,
    model: Model,
    slab_filter: Optional[Callable[[Slab], bool]] = None,
    client: Client = DEFAULT_CLIENT,
    lifetime: Lifetime = Lifetime.SAVE,
) -> List[AdsorbateSlabRelaxation]:
    """
    Search for adsorbate binding sites on surfaces of a bulk material.
    This executes the following steps:

        1. Ensure that both the adsorbate and bulk are supported in the
           OCP API.
        2. Enumerate unique surfaces from the bulk material. If a
           slab_filter function is provided, only those surfaces for
           which the filter returns True will be kept.
        3. Enumerate likely binding sites for the input adsorbate on each
           of the generated surfaces.
        4. Relax each generated surface+adsorbate structure by refining
           atomic positions to minimize forces generated by the input model.

    Args:
        adsorbate: SMILES string describing the adsorbate to place.
        bulk: The ID (typically Materials Project MP ID) of the bulk material
            on which the adsorbate will be placed.
        model: The type of the model to use when calculating forces during
            relaxations.
        slab_filter: If not None, a function that filters which generated
            slabs will be considered when placing adsorbates.
        client: The OCP API client to use.
        lifetime: Whether relaxations should be saved on the server, be marked
            as ephemeral (allowing them to deleted in the future), or deleted
            immediately.

    Returns:
        Details of each adsorbate binding site, including results of relaxing
        to locally-optimized positions using the input model.

    Raises:
        UnsupportedBulkException if the requested bulk is not supported.
        UnsupportedAdsorbateException if the requested adsorbate is not
            supported.
    """
    with set_context_var(_CTX_AD_BULK, (adsorbate, bulk)):
        # Make sure the input adsorbate is supported in the API
        log.info(f"Ensuring that adsorbate {adsorbate} is supported")
        await _ensure_adsorbate_supported(
            client=client,
            adsorbate=adsorbate,
        )

        # Make sure the input bulk is supported in the API
        log.info(f"Ensuring that bulk {bulk} is supported")
        bulk_obj: Bulk = await _get_bulk_if_supported(
            client=client,
            bulk=bulk,
        )

        # Fetch all slabs for the bulk
        log.info("Generating surfaces")
        slabs: List[Slab] = await _get_slabs(
            client=client,
            bulk=bulk_obj,
            slab_filter=slab_filter,
        )

        # Generate adsorbate placements and run relaxations
        tasks: List[asyncio.Task] = [
            asyncio.create_task(
                _find_binding_sites_on_slab(
                    client=client,
                    adsorbate=adsorbate,
                    bulk=bulk_obj,
                    slab=slab,
                    model=model,
                    lifetime=lifetime,
                )
            )
            for slab in slabs
        ]
        _, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        # There shouldn't be any pending tasks since we wait for all coroutines
        # to finish above. However, make sure they are cancelled in case there
        # is some edge case in which tasks are still running.
        for t in pending:
            log.warning("Cancelling running task")
            with suppress(asyncio.CancelledError):
                t.cancel()
                await t

        results: List[AdsorbateSlabRelaxation] = []
        for t in tasks:
            results.extend(t.result())
        return results


def keep_slabs_with_miller_indices(
    allowed_miller_indices: Iterable[Tuple[int, int, int]],
) -> Callable[[Slab], bool]:
    """
    Filter that can be passed to find_adsorbate_binding_sites that keeps
    surfaces for which the Miller Indices match one set in the input list.

    Args:
        allowed_miller_indices: The list of Miller Indices that will be kept.
            Any surfaces for which the Miller Indices are not in the input
            list will be discarded.

    Returns:
        The function that filters surfaces, keeping only those in the input
        list.
    """
    unique_millers: Set[Tuple[int, int, int]] = set(allowed_miller_indices)

    def func(slab: Slab) -> bool:
        return slab.metadata.millers in unique_millers

    return func
