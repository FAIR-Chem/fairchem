import asyncio
import json
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

import requests

from .models import (
    Adsorbates,
    AdsorbateSlabConfigs,
    AdsorbateSlabRelaxationsRequest,
    AdsorbateSlabRelaxationsResults,
    AdsorbateSlabRelaxationsSystem,
    Atoms,
    Bulk,
    Bulks,
    Models,
    Slab,
    Slabs,
)


class RequestException(Exception):
    """
    Exception raised any time there is an error while making an API call.
    """

    def __init__(self, method: str, url: str, cause: str) -> None:
        """
        Args:
            method: The type of the method being run (POST, GET, etc.).
            url: The full URL that was called.
            cause: A description of the failure.
        """
        super().__init__(f"Request to {method} {url} failed. {cause}")


class NonRetryableRequestException(RequestException):
    """
    Exception raised when an API call is rejected for a reason that will
    not succeed on retry. For example, this might include a malformed request
    or action that is not allowed.
    """

    def __init__(self, method: str, url: str, cause: str) -> None:
        """
        Args:
            method: The type of the method being run (POST, GET, etc.).
            url: The full URL that was called.
            cause: A description of the failure.
        """
        super().__init__(method=method, url=url, cause=cause)


class RateLimitExceededException(RequestException):
    """
    Exception raised when an API call is rejected because a rate limit has
    been exceeded.

    Attributes:
        retry_after: If known, the time to wait before the next attempt to
            call the API should be made.
    """

    def __init__(
        self,
        method: str,
        url: str,
        retry_after: Optional[timedelta] = None,
    ) -> None:
        """
        Args:
            method: The type of the method being run (POST, GET, etc.).
            url: The full URL that was called.
            retry_after: If known, the time to wait before the next attempt
                to call the API should be made.
        """
        super().__init__(method=method, url=url, cause="Exceeded rate limit")
        self.retry_after: Optional[timedelta] = retry_after


class Client:
    """
    Exposes each route in the OCP API as a method.
    """

    def __init__(
        self,
        host: str = "open-catalyst-api.metademolab.com",
        scheme: str = "https",
    ) -> None:
        """
        Args:
            host: The host that will be called.
            scheme: The scheme used when making API calls.
        """
        self._host = host
        self._base_url = f"{scheme}://{host}"

    @property
    def host(self) -> str:
        """
        The host being called by this client.
        """
        return self._host

    async def get_models(self) -> Models:
        """
        Fetch the list of models that are supported in the API.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            The models that are supported in the API.
        """
        response: str = await self._run_request(
            path="ocp/models",
            method="GET",
        )
        return Models.from_json(response)

    async def get_bulks(self) -> Bulks:
        """
        Fetch the list of bulk materials that are supported in the API.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            The bulks that are supported throughout the API.
        """
        response: str = await self._run_request(
            path="ocp/bulks",
            method="GET",
        )
        return Bulks.from_json(response)

    async def get_adsorbates(self) -> Adsorbates:
        """
        Fetch the list of adsorbates that are supported in the API.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            The adsorbates that are supported throughout the API.
        """
        response: str = await self._run_request(
            path="ocp/adsorbates",
            method="GET",
        )
        return Adsorbates.from_json(response)

    async def get_slabs(self, bulk: Union[str, Bulk]) -> Slabs:
        """
        Get a unique list of slabs for the input bulk structure.

        Args:
            bulk: If a string, the id of the bulk to use. Otherwise the Bulk
                instance to use.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            Slabs for each of the unique surfaces of the material.
        """
        response: str = await self._run_request(
            path="ocp/slabs",
            method="POST",
            data=json.dumps(
                {"bulk_src_id": bulk.src_id if isinstance(bulk, Bulk) else bulk}
            ),
            headers={"Content-Type": "application/json"},
        )
        return Slabs.from_json(response)

    async def get_adsorbate_slab_configs(
        self, adsorbate: str, slab: Slab
    ) -> AdsorbateSlabConfigs:
        """
        Get a list of possible binding sites for the input adsorbate on the
        input slab.

        Args:
            adsorbate: Description of the the adsorbate to place.
            slab: Information about the slab on which the adsorbate should
                be placed.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            Configurations for each adsorbate binding site on the slab.
        """
        response: str = await self._run_request(
            path="ocp/adsorbate-slab-configs",
            method="POST",
            data=json.dumps(
                {
                    "adsorbate": adsorbate,
                    "slab": slab.to_dict(),
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        return AdsorbateSlabConfigs.from_json(response)

    async def submit_adsorbate_slab_relaxations(
        self,
        adsorbate: str,
        adsorbate_configs: List[Atoms],
        bulk: Bulk,
        slab: Slab,
        model: str,
        ephemeral: bool = False,
    ) -> AdsorbateSlabRelaxationsSystem:
        """
        Starts relaxations of the input adsorbate configurations on the input
        slab using energies and forces returned by the input model. Relaxations
        are run asynchronously and results can be fetched using the system id
        that is returned from this method.

        Args:
            adsorbate: Description of the adsorbate being simulated.
            adsorbate_configs: List of adsorbate configurations to relax. This
                should only include the adsorbates themselves; the surface is
                defined in the "slab" field that is a peer to this one.
            bulk: Details of the bulk material being simulated.
            slab: The structure of the slab on which adsorbates are placed.
            model: The model that will be used to evaluate energies and forces
                during relaxations.
            ephemeral: If False (default), any later attempt to delete the
                generated relaxations will be rejected. If True, deleting the
                relaxations will be allowed, which is generally useful for
                testing when there is no reason for results to be persisted.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            IDs of the relaxations.
        """
        response: str = await self._run_request(
            path="ocp/adsorbate-slab-relaxations",
            method="POST",
            data=json.dumps(
                {
                    "adsorbate": adsorbate,
                    "adsorbate_configs": [a.to_dict() for a in adsorbate_configs],
                    "bulk": bulk.to_dict(),
                    "slab": slab.to_dict(),
                    "model": model,
                    "ephemeral": ephemeral,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        return AdsorbateSlabRelaxationsSystem.from_json(response)

    async def get_adsorbate_slab_relaxations_request(
        self, system_id: str
    ) -> AdsorbateSlabRelaxationsRequest:
        """
        Fetches the original relaxations request for the input system.

        Args:
            system_id: The ID of the system to fetch.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            The original request that was made when submitting relaxations.
        """
        response: str = await self._run_request(
            path=f"ocp/adsorbate-slab-relaxations/{system_id}",
            method="GET",
        )
        return AdsorbateSlabRelaxationsRequest.from_json(response)

    async def get_adsorbate_slab_relaxations_results(
        self,
        system_id: str,
        config_ids: Optional[List[int]] = None,
        fields: Optional[List[str]] = None,
    ) -> AdsorbateSlabRelaxationsResults:
        """
        Fetches relaxation results for the input system.

        Args:
            system_id: The system id of the relaxations.
            config_ids: If defined and not empty, a subset of configurations
                to fetch. Otherwise all configurations are returned.
            fields: If defined and not empty, a subset of fields in each
                configuration to fetch. Otherwise all fields are returned.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            The relaxation results for each configuration in the system.
        """
        params: Dict[str, Any] = {}
        if fields:
            params["field"] = fields
        if config_ids:
            params["config_id"] = config_ids
        response: str = await self._run_request(
            path=f"ocp/adsorbate-slab-relaxations/{system_id}/configs",
            method="GET",
            params=params,
        )
        return AdsorbateSlabRelaxationsResults.from_json(response)

    async def delete_adsorbate_slab_relaxations(self, system_id: str) -> None:
        """
        Deletes all relaxation results for the input system.

        Args:
            system_id: The ID of the system to delete.

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.
        """
        await self._run_request(
            path=f"ocp/adsorbate-slab-relaxations/{system_id}",
            method="DELETE",
        )

    async def _run_request(self, path: str, method: str, **kwargs) -> str:
        """
        Helper method that runs the input request on a thread so that
        it doesn't block the event loop on the calling thread.

        Args:
            path: The URL path to make the request against.
            method: The HTTP method to use (GET, POST, etc.).

        Raises:
            RateLimitExceededException: If the call was rejected because a
                server side rate limit was breached.
            NonRetryableRequestException: If the call was rejected and a retry
                is not expected to succeed.
            RequestException: For all other errors when making the request; it
                is possible, though not guaranteed, that a retry could succeed.

        Returns:
            The response body from the request as a string.
        """

        # Make the request
        url = f"{self._base_url}/{path}"
        try:
            response: requests.Response = await asyncio.to_thread(
                requests.request,
                method=method,
                url=url,
                **kwargs,
            )
        except Exception as e:
            raise RequestException(
                method=method,
                url=url,
                cause=f"Exception while making request: {type(e).__name__}: {e}",
            ) from e

        # Check the response code
        if response.status_code >= 300:
            # Exceeded server side rate limit
            if response.status_code == 429:
                retry_after: Optional[str] = response.headers.get("Retry-After", None)
                raise RateLimitExceededException(
                    method=method,
                    url=url,
                    retry_after=timedelta(seconds=float(retry_after))
                    if retry_after is not None
                    else None,
                )

            # Treat all other 400-level response codes as ones that are
            # unlikely to succeed on retry
            cause: str = (
                f"Unexpected response code: {response.status_code}. "
                f"Response body: {response.text}"
            )
            if response.status_code >= 400 and response.status_code < 500:
                raise NonRetryableRequestException(
                    method=method,
                    url=url,
                    cause=cause,
                )

            # Treat all other errors as ones that might succeed on retry
            raise RequestException(
                method=method,
                url=url,
                cause=cause,
            )

        return response.text
