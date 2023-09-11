import asyncio
import json
from typing import Union

import requests

from ocpapi.models import (
    AdsorbateSlabConfigsResponse,
    AdsorbatesResponse,
    Bulk,
    BulksResponse,
    Slab,
    SlabsResponse,
)


class RequestException(Exception):
    def __init__(self, method: str, url: str, cause: str) -> None:
        super().__init__(f"Request to {method} {url} failed. {cause}")


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

    async def get_bulks(self) -> BulksResponse:
        """
        Fetch the list of bulk materials that are supported in the API.

        Returns:
            BulksResponse
        """
        response = await self._run_request(
            url=f"{self._base_url}/bulks",
            method="GET",
            expected_response_code=200,
        )
        return BulksResponse.from_json(response)

    async def get_adsorbates(self) -> AdsorbatesResponse:
        """
        Fetch the list of adsorbates that are supported in the API.

        Returns:
            AdsorbatesResponse
        """
        response = await self._run_request(
            url=f"{self._base_url}/adsorbates",
            method="GET",
            expected_response_code=200,
        )
        return AdsorbatesResponse.from_json(response)

    async def get_slabs(self, bulk: Union[str, Bulk]) -> SlabsResponse:
        """
        Get a unique list of slabs for for the input bulk structure.

        Args:
            bulk: If a string, the id of the bulk to use. Otherwise the Bulk
                instance to use.

        Returns:
            SlabsResponse
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
        return SlabsResponse.from_json(response)

    async def get_adsorbate_slab_configs(
        self, adsorbate: str, slab: Slab
    ) -> AdsorbateSlabConfigsResponse:
        """
        Get a list of possible binding sites for the input adsorbate on the
        input slab.

        Args:
            adsorbate: SMILES string describing the adsorbate to place.
            slab: Information about the slab on which the adsorbate should
                be placed.

        Returns:
            AdsorbateSlabConfigsResponse
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
        return AdsorbateSlabConfigsResponse.from_json(response)

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
