import logging
from contextlib import contextmanager
from typing import Generator
from unittest import IsolatedAsyncioTestCase, mock

import requests

from ocpapi.client import Client, Model
from ocpapi.models import Atoms, Bulk, Slab, SlabMetadata

log = logging.getLogger(__name__)


@contextmanager
def _try_ensure_system_deleted(
    base_url: str,
    system_id: str,
) -> Generator[None, None, None]:
    """
    Immediately yields control to the caller. When control returns to this
    function, try to delete the system with the input id.
    """
    try:
        yield
    finally:
        try:
            requests.delete(f"{base_url}/adsorbate-slab-relaxations/{system_id}")
            log.info(f"Deleted system with id={system_id}")
        except BaseException:
            log.exception(f"Failed to delete system with id={system_id}")


class TestClient(IsolatedAsyncioTestCase):
    """
    Tests that calls to a real server are handled correctly.
    """

    TEST_HOST = "https://open-catalyst-api.metademolab.com/ocp"

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

    async def test_get_adsorbate_slab_configs(self) -> None:
        # Make sure that adsorbate placements are generated for a slab
        # and adsorbate combination that is known to be supported

        client = Client(self.TEST_HOST)
        response = await client.get_adsorbate_slab_configs(
            adsorbate="*CO",
            slab=Slab(
                atoms=Atoms(
                    cell=(
                        (11.6636, 0, 0),
                        (-5.8318, 10.1010, 0),
                        (0, 0, 38.0931),
                    ),
                    pbc=(True, True, True),
                    numbers=[14] * 54,
                    tags=[0] * 54,
                    positions=[
                        (1.9439, 1.1223, 17.0626),
                        (-0.0, 0.0, 20.237),
                        (-0.0, 2.2447, 23.4114),
                        (1.9439, 1.1223, 14.6817),
                        (3.8879, 0.0, 17.8562),
                        (-0.0, 2.2447, 21.0306),
                        (-0.0, 4.4893, 17.0626),
                        (-1.9439, 3.367, 20.237),
                        (-1.9439, 5.6117, 23.4114),
                        (-0.0, 4.4893, 14.6817),
                        (1.9439, 3.367, 17.8562),
                        (-1.9439, 5.6117, 21.0306),
                        (-1.9439, 7.8563, 17.0626),
                        (-3.8879, 6.734, 20.237),
                        (-3.8879, 8.9786, 23.4114),
                        (-1.9439, 7.8563, 14.6817),
                        (-0.0, 6.734, 17.8562),
                        (-3.8879, 8.9786, 21.0306),
                        (5.8318, 1.1223, 17.0626),
                        (3.8879, 0.0, 20.237),
                        (3.8879, 2.2447, 23.4114),
                        (5.8318, 1.1223, 14.6817),
                        (7.7757, 0.0, 17.8562),
                        (3.8879, 2.2447, 21.0306),
                        (3.8879, 4.4893, 17.0626),
                        (1.9439, 3.367, 20.237),
                        (1.9439, 5.6117, 23.4114),
                        (3.8879, 4.4893, 14.6817),
                        (5.8318, 3.367, 17.8562),
                        (1.9439, 5.6117, 21.0306),
                        (1.9439, 7.8563, 17.0626),
                        (-0.0, 6.734, 20.237),
                        (-0.0, 8.9786, 23.4114),
                        (1.9439, 7.8563, 14.6817),
                        (3.8879, 6.734, 17.8562),
                        (-0.0, 8.9786, 21.0306),
                        (9.7197, 1.1223, 17.0626),
                        (7.7757, 0.0, 20.237),
                        (7.7757, 2.2447, 23.4114),
                        (9.7197, 1.1223, 14.6817),
                        (11.6636, 0.0, 17.8562),
                        (7.7757, 2.2447, 21.0306),
                        (7.7757, 4.4893, 17.0626),
                        (5.8318, 3.367, 20.237),
                        (5.8318, 5.6117, 23.4114),
                        (7.7757, 4.4893, 14.6817),
                        (9.7197, 3.367, 17.8562),
                        (5.8318, 5.6117, 21.0306),
                        (5.8318, 7.8563, 17.0626),
                        (3.8879, 6.734, 20.237),
                        (3.8879, 8.9786, 23.4114),
                        (5.8318, 7.8563, 14.6817),
                        (7.7757, 6.734, 17.8562),
                        (3.8879, 8.9786, 21.0306),
                    ],
                ),
                metadata=SlabMetadata(
                    bulk_src_id="mp-149",
                    millers=(1, 1, 1),
                    shift=0.125,
                    top=True,
                ),
            ),
        )

        self.assertGreater(len(response.adsorbate_configs), 10)

    async def test_submit_adsorbate_slab_relaxations__gemnet_oc(self) -> None:
        # Make sure that a relaxation can be started for an adsorbate
        # placement on a slab with the gemnet oc model

        client = Client(self.TEST_HOST)
        response = await client.submit_adsorbate_slab_relaxations(
            adsorbate="*CO",
            adsorbate_configs=[
                Atoms(
                    cell=(
                        (11.6636, 0, 0),
                        (-5.8318, 10.1010, 0),
                        (0, 0, 38.0931),
                    ),
                    pbc=(True, True, False),
                    numbers=[6, 8],
                    tags=[2, 2],
                    positions=[
                        (1.9439, 3.3670, 22.2070),
                        (1.9822, 3.2849, 23.3697),
                    ],
                )
            ],
            bulk=Bulk(src_id="mp-149", elements=["Si"], formula="Si"),
            slab=Slab(
                atoms=Atoms(
                    cell=(
                        (11.6636, 0, 0),
                        (-5.8318, 10.1010, 0),
                        (0, 0, 38.0931),
                    ),
                    pbc=(True, True, True),
                    numbers=[14] * 54,
                    tags=[0] * 54,
                    positions=[
                        (1.9439, 1.1223, 17.0626),
                        (-0.0, 0.0, 20.237),
                        (-0.0, 2.2447, 23.4114),
                        (1.9439, 1.1223, 14.6817),
                        (3.8879, 0.0, 17.8562),
                        (-0.0, 2.2447, 21.0306),
                        (-0.0, 4.4893, 17.0626),
                        (-1.9439, 3.367, 20.237),
                        (-1.9439, 5.6117, 23.4114),
                        (-0.0, 4.4893, 14.6817),
                        (1.9439, 3.367, 17.8562),
                        (-1.9439, 5.6117, 21.0306),
                        (-1.9439, 7.8563, 17.0626),
                        (-3.8879, 6.734, 20.237),
                        (-3.8879, 8.9786, 23.4114),
                        (-1.9439, 7.8563, 14.6817),
                        (-0.0, 6.734, 17.8562),
                        (-3.8879, 8.9786, 21.0306),
                        (5.8318, 1.1223, 17.0626),
                        (3.8879, 0.0, 20.237),
                        (3.8879, 2.2447, 23.4114),
                        (5.8318, 1.1223, 14.6817),
                        (7.7757, 0.0, 17.8562),
                        (3.8879, 2.2447, 21.0306),
                        (3.8879, 4.4893, 17.0626),
                        (1.9439, 3.367, 20.237),
                        (1.9439, 5.6117, 23.4114),
                        (3.8879, 4.4893, 14.6817),
                        (5.8318, 3.367, 17.8562),
                        (1.9439, 5.6117, 21.0306),
                        (1.9439, 7.8563, 17.0626),
                        (-0.0, 6.734, 20.237),
                        (-0.0, 8.9786, 23.4114),
                        (1.9439, 7.8563, 14.6817),
                        (3.8879, 6.734, 17.8562),
                        (-0.0, 8.9786, 21.0306),
                        (9.7197, 1.1223, 17.0626),
                        (7.7757, 0.0, 20.237),
                        (7.7757, 2.2447, 23.4114),
                        (9.7197, 1.1223, 14.6817),
                        (11.6636, 0.0, 17.8562),
                        (7.7757, 2.2447, 21.0306),
                        (7.7757, 4.4893, 17.0626),
                        (5.8318, 3.367, 20.237),
                        (5.8318, 5.6117, 23.4114),
                        (7.7757, 4.4893, 14.6817),
                        (9.7197, 3.367, 17.8562),
                        (5.8318, 5.6117, 21.0306),
                        (5.8318, 7.8563, 17.0626),
                        (3.8879, 6.734, 20.237),
                        (3.8879, 8.9786, 23.4114),
                        (5.8318, 7.8563, 14.6817),
                        (7.7757, 6.734, 17.8562),
                        (3.8879, 8.9786, 21.0306),
                    ],
                ),
                metadata=SlabMetadata(
                    bulk_src_id="mp-149",
                    millers=(1, 1, 1),
                    shift=0.125,
                    top=True,
                ),
            ),
            model=Model.GEMNET_OC_BASE_S2EF_ALL_MD,
            ephemeral=True,
        )

        with _try_ensure_system_deleted(self.TEST_HOST, response.system_id):
            self.assertNotEqual(response.system_id, "")
            self.assertEqual(len(response.config_ids), 1)

    async def test_submit_adsorbate_slab_relaxations__equiformer_v2(self) -> None:
        # Make sure that a relaxation can be started for an adsorbate
        # placement on a slab with the equiformer v2 model

        client = Client(self.TEST_HOST)
        response = await client.submit_adsorbate_slab_relaxations(
            adsorbate="*CO",
            adsorbate_configs=[
                Atoms(
                    cell=(
                        (11.6636, 0, 0),
                        (-5.8318, 10.1010, 0),
                        (0, 0, 38.0931),
                    ),
                    pbc=(True, True, False),
                    numbers=[6, 8],
                    tags=[2, 2],
                    positions=[
                        (1.9439, 3.3670, 22.2070),
                        (1.9822, 3.2849, 23.3697),
                    ],
                )
            ],
            bulk=Bulk(src_id="mp-149", elements=["Si"], formula="Si"),
            slab=Slab(
                atoms=Atoms(
                    cell=(
                        (11.6636, 0, 0),
                        (-5.8318, 10.1010, 0),
                        (0, 0, 38.0931),
                    ),
                    pbc=(True, True, True),
                    numbers=[14] * 54,
                    tags=[0] * 54,
                    positions=[
                        (1.9439, 1.1223, 17.0626),
                        (-0.0, 0.0, 20.237),
                        (-0.0, 2.2447, 23.4114),
                        (1.9439, 1.1223, 14.6817),
                        (3.8879, 0.0, 17.8562),
                        (-0.0, 2.2447, 21.0306),
                        (-0.0, 4.4893, 17.0626),
                        (-1.9439, 3.367, 20.237),
                        (-1.9439, 5.6117, 23.4114),
                        (-0.0, 4.4893, 14.6817),
                        (1.9439, 3.367, 17.8562),
                        (-1.9439, 5.6117, 21.0306),
                        (-1.9439, 7.8563, 17.0626),
                        (-3.8879, 6.734, 20.237),
                        (-3.8879, 8.9786, 23.4114),
                        (-1.9439, 7.8563, 14.6817),
                        (-0.0, 6.734, 17.8562),
                        (-3.8879, 8.9786, 21.0306),
                        (5.8318, 1.1223, 17.0626),
                        (3.8879, 0.0, 20.237),
                        (3.8879, 2.2447, 23.4114),
                        (5.8318, 1.1223, 14.6817),
                        (7.7757, 0.0, 17.8562),
                        (3.8879, 2.2447, 21.0306),
                        (3.8879, 4.4893, 17.0626),
                        (1.9439, 3.367, 20.237),
                        (1.9439, 5.6117, 23.4114),
                        (3.8879, 4.4893, 14.6817),
                        (5.8318, 3.367, 17.8562),
                        (1.9439, 5.6117, 21.0306),
                        (1.9439, 7.8563, 17.0626),
                        (-0.0, 6.734, 20.237),
                        (-0.0, 8.9786, 23.4114),
                        (1.9439, 7.8563, 14.6817),
                        (3.8879, 6.734, 17.8562),
                        (-0.0, 8.9786, 21.0306),
                        (9.7197, 1.1223, 17.0626),
                        (7.7757, 0.0, 20.237),
                        (7.7757, 2.2447, 23.4114),
                        (9.7197, 1.1223, 14.6817),
                        (11.6636, 0.0, 17.8562),
                        (7.7757, 2.2447, 21.0306),
                        (7.7757, 4.4893, 17.0626),
                        (5.8318, 3.367, 20.237),
                        (5.8318, 5.6117, 23.4114),
                        (7.7757, 4.4893, 14.6817),
                        (9.7197, 3.367, 17.8562),
                        (5.8318, 5.6117, 21.0306),
                        (5.8318, 7.8563, 17.0626),
                        (3.8879, 6.734, 20.237),
                        (3.8879, 8.9786, 23.4114),
                        (5.8318, 7.8563, 14.6817),
                        (7.7757, 6.734, 17.8562),
                        (3.8879, 8.9786, 21.0306),
                    ],
                ),
                metadata=SlabMetadata(
                    bulk_src_id="mp-149",
                    millers=(1, 1, 1),
                    shift=0.125,
                    top=True,
                ),
            ),
            model=Model.EQUIFORMER_V2_31M_S2EF_ALL_MD,
            ephemeral=True,
        )

        with _try_ensure_system_deleted(self.TEST_HOST, response.system_id):
            self.assertNotEqual(response.system_id, "")
            self.assertEqual(len(response.config_ids), 1)
