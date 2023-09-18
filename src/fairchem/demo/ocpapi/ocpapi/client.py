import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests

from ocpapi.models import (
    AdsorbateSlabConfigs,
    AdsorbateSlabRelaxationsSystem,
    AdsorbateSlabRelaxationsResults,
    Adsorbates,
    Atoms,
    Bulk,
    Bulks,
    Slab,
    Slabs,
)


class RequestException(Exception):
    def __init__(self, method: str, url: str, cause: str) -> None:
        super().__init__(f"Request to {method} {url} failed. {cause}")


class Model(Enum):
    """
    ML model that can be used in adsorbate-slab relaxations.

    Attributes:
        GEMNET_OC_BASE_S2EF_ALL_MD: https://arxiv.org/abs/2204.02782
        EQUIFORMER_V2_31M_S2EF_ALL_MD: https://arxiv.org/abs/2306.12059
    """

    GEMNET_OC_BASE_S2EF_ALL_MD = "gemnet_oc_base_s2ef_all_md"
    EQUIFORMER_V2_31M_S2EF_ALL_MD = "equiformer_v2_31M_s2ef_all_md"

    def __str__(self) -> str:
        return self.value


class Client:
    """
    Exposes each route in the OCP API as a method.
    """

    def __init__(
        self,
        base_url: str = "https://open-catalyst-api.metademolab.com/ocp/",
    ) -> None:
        """
        Args:
            base_url: The base URL for all API requests.
        """
        # Normalize the base URL so that all methods below can assume it
        # does not end in a '/' character
        self._base_url = base_url.rstrip("/")

    async def get_bulks(self) -> Bulks:
        """
        Fetch the list of bulk materials that are supported in the API.

        Returns:
            Bulks
        """
        response = await self._run_request(
            url=f"{self._base_url}/bulks",
            method="GET",
            expected_response_code=200,
        )
        return Bulks.from_json(response)

    async def get_adsorbates(self) -> Adsorbates:
        """
        Fetch the list of adsorbates that are supported in the API.

        Returns:
            Adsorbates
        """
        response = await self._run_request(
            url=f"{self._base_url}/adsorbates",
            method="GET",
            expected_response_code=200,
        )
        return Adsorbates.from_json(response)

    async def get_slabs(self, bulk: Union[str, Bulk]) -> Slabs:
        """
        Get a unique list of slabs for the input bulk structure.

        Args:
            bulk: If a string, the id of the bulk to use. Otherwise the Bulk
                instance to use.

        Returns:
            Slabs
        """
        response = await self._run_request(
            url=f"{self._base_url}/slabs",
            method="POST",
            expected_response_code=200,
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
            adsorbate: SMILES string describing the adsorbate to place.
            slab: Information about the slab on which the adsorbate should
                be placed.

        Returns:
            AdsorbateSlabConfigs
        """
        response = await self._run_request(
            url=f"{self._base_url}/adsorbate-slab-configs",
            method="POST",
            expected_response_code=200,
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
        model: Union[Model, str],
        ephemeral: bool = False,
    ) -> AdsorbateSlabRelaxationsSystem:
        """
        Starts relaxations of the input adsorbate configurations on the input
        slab using energies and forces returned by the input model. Relaxations
        are run asynchronously and results can be fetched using the system id
        that is returned from this method.

        Args:
            adsorbate: SMILES string describing the adsorbate being simulated.
            adsorbate_configs: List of adsorbate configurations to relax. This
                should only include the adsorbates themselves; the surface is
                defined in the "slab" field that is a peer to this one.
            bulk: Details of the bulk material being simulated.
            slab: The structure of the slab on which adsorbates are placed.
            model: The model that will be used to evaluate energies and forces
                during relaxations. Prefer using the enumerated Model values,
                but a free-form string can be supplied if this client version
                does not support a model known to exist on the API being
                invoked.
            ephemeral: If False (default), any later attempt to delete the
                generated relaxations will be rejected. If True, deleting the
                relaxations will be allowed, which is generally useful for
                testing when there is no reason for results to be persisted.

        Returns:
            AdsorbateSlabRelaxationsSystem
        """
        response = await self._run_request(
            url=f"{self._base_url}/adsorbate-slab-relaxations",
            method="POST",
            expected_response_code=200,
            data=json.dumps(
                {
                    "adsorbate": adsorbate,
                    "adsorbate_configs": [a.to_dict() for a in adsorbate_configs],
                    "bulk": bulk.to_dict(),
                    "slab": slab.to_dict(),
                    "model": str(model),
                    "ephemeral": ephemeral,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        return AdsorbateSlabRelaxationsSystem.from_json(response)

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

        Returns:
            AdsorbateSlabRelaxationsResults
        """
        params: Dict[str, Any] = {}
        if fields:
            params["field"] = fields
        if config_ids:
            params["config_id"] = config_ids
        response = await self._run_request(
            url=f"{self._base_url}/adsorbate-slab-relaxations/{system_id}/configs",
            method="GET",
            expected_response_code=200,
            params=params,
        )
        return AdsorbateSlabRelaxationsResults.from_json(response)

    async def delete_adsorbate_slab_relaxations(self, system_id: str) -> None:
        """
        Deletes all relaxation results for the input system.

        Args:
            system_id: The ID of the system to delete.
        """
        await self._run_request(
            url=f"{self._base_url}/adsorbate-slab-relaxations/{system_id}",
            method="DELETE",
            expected_response_code=200,
        )

    async def _run_request(
        self, url: str, method: str, expected_response_code: int, **kwargs
    ) -> str:
        """
        Helper method that runs the input request on a thread so that
        it doesn't block the event loop on the calling thread.

        Args:
            url: The full URL to make the request against.
            method: The HTTP method to use (GET, POST, etc.).
            expected_response_code: The response code that indicates success.

        Returns:
            The response body from the request as a string.
        """

        # Make the request
        try:
            response = await asyncio.to_thread(
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
        if response.status_code != expected_response_code:
            raise RequestException(
                method=method,
                url=url,
                cause=(
                    f"Expected response code {expected_response_code}; "
                    f"got {response.status_code}. Body = {response.text}"
                ),
            )

        return response.text
