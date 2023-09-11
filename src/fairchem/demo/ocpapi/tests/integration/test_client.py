from unittest import IsolatedAsyncioTestCase, mock

from ocpapi.client import Client
from ocpapi.models import Bulk, Slab, SlabMetadata


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

    async def test_get_slabs(self) -> None:
        # Make sure that at least one of the expected slabs is in the response

        client = Client(self.TEST_HOST)
        response = await client.get_slabs("mp-149")

        self.assertIn(
            Slab(
                # Don't worry about checking the specific values in the
                # returned structure. This could be unstable if the code
                # on the server changes and we don't necessarily care here
                # what each value is.
                atoms=mock.ANY,
                metadata=SlabMetadata(
                    bulk_src_id="mp-149",
                    millers=(1, 1, 1),
                    shift=0.125,
                    top=True,
                ),
            ),
            response.slabs,
        )
