from unittest import IsolatedAsyncioTestCase

from ocpapi.client import Client
from ocpapi.models import Bulk


class TestClient(IsolatedAsyncioTestCase):
    """
    Tests that calls to a real server are handled correctly.
    """

    TEST_HOST = "https://open-catalyst-api.metademolab.com/ocp/"

    async def test_get_bulks(self) -> None:
        # Make sure that at least one of the expected bulks is in the response

        client = Client(self.TEST_HOST)
        response = await client.get_bulks()

        self.assertIn(
            Bulk(src_id="mp-149", elements=["Si"], formula="Si"),
            response.bulks_supported,
        )

    async def test_get_adsorbates(self) -> None:
        # Make sure that at least one of the expected adsorbates is in the
        # response

        client = Client(self.TEST_HOST)
        response = await client.get_adsorbates()

        self.assertIn("*CO", response.adsorbates_supported)
