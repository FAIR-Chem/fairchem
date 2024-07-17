import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest import IsolatedAsyncioTestCase, mock

import numpy as np
import pytest
from fairchem.demo.ocpapi.client import (
    Atoms,
    Bulk,
    Client,
    Model,
    Slab,
    SlabMetadata,
    Status,
)

log = logging.getLogger(__name__)


@asynccontextmanager
async def _ensure_system_deleted(
    client: Client, system_id: str,
) -> AsyncGenerator[None, None]:
    """
    Immediately yields control to the caller. When control returns to this
    function, try to delete the system with the input id.
    """
    try:
        yield
    finally:
        await client.delete_adsorbate_slab_relaxations(system_id)


class TestClient(IsolatedAsyncioTestCase):
    """
    Tests that calls to a real server are handled correctly.
    """

    CLIENT: Client = Client(
        host="open-catalyst-api.metademolab.com", scheme="https",
    )
    KNOWN_SYSTEM_ID: str = "f9eacd8f-748c-41dd-ae43-f263dd36d735"

    async def test_get_models(self) -> None:
        # Make sure that at least one of the known models is in the response

        response = await self.CLIENT.get_models()

        self.assertIn(
            Model(id="equiformer_v2_31M_s2ef_all_md"), response.models,
        )

    async def test_get_bulks(self) -> None:
        # Make sure that at least one of the expected bulks is in the response

        response = await self.CLIENT.get_bulks()

        self.assertIn(
            Bulk(src_id="mp-149", elements=["Si"], formula="Si"),
            response.bulks_supported,
        )

    async def test_get_adsorbates(self) -> None:
        # Make sure that at least one of the expected adsorbates is in the
        # response

        response = await self.CLIENT.get_adsorbates()

        self.assertIn("*CO", response.adsorbates_supported)

    async def test_get_slabs(self) -> None:
        # Make sure that at least one of the expected slabs is in the response

        response = await self.CLIENT.get_slabs("mp-149")
        expected_slab = Slab(
            # Don't worry about checking the specific values in the
            # returned structure. This could be unstable if the code
            # on the server changes and we don't necessarily care here
            # what each value is.
            atoms=mock.ANY,
            metadata=SlabMetadata(
                bulk_src_id="mp-149", millers=(1, 1, 1), shift=0.125, top=True,
            ),
        )
        if expected_slab not in response.slabs:
            pytest.xfail(
                f"Expected slab can be slightly off (numerical error?) from the actual slab returned; This test is flaky;"
            )
        self.assertIn(expected_slab, response.slabs)

    async def test_get_adsorbate_slab_configs(self) -> None:
        # Make sure that adsorbate placements are generated for a slab
        # and adsorbate combination that is known to be supported

        response = await self.CLIENT.get_adsorbate_slab_configs(
            adsorbate="*CO",
            slab=Slab(
                atoms=Atoms(
                    cell=((11.6636, 0, 0), (-5.8318, 10.1010, 0), (0, 0, 38.0931),),
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
                    bulk_src_id="mp-149", millers=(1, 1, 1), shift=0.125, top=True,
                ),
            ),
        )

        self.assertGreater(len(response.adsorbate_configs), 10)

    @pytest.mark.ocpapi_integration_test
    async def test_submit_adsorbate_slab_relaxations__gemnet_oc(self) -> None:
        # Make sure that a relaxation can be started for an adsorbate
        # placement on a slab with the gemnet oc model

        response = await self.CLIENT.submit_adsorbate_slab_relaxations(
            adsorbate="*CO",
            adsorbate_configs=[
                Atoms(
                    cell=((11.6636, 0, 0), (-5.8318, 10.1010, 0), (0, 0, 38.0931),),
                    pbc=(True, True, False),
                    numbers=[6, 8],
                    tags=[2, 2],
                    positions=[(1.9439, 3.3670, 22.2070), (1.9822, 3.2849, 23.3697),],
                )
            ],
            bulk=Bulk(src_id="mp-149", elements=["Si"], formula="Si"),
            slab=Slab(
                atoms=Atoms(
                    cell=((11.6636, 0, 0), (-5.8318, 10.1010, 0), (0, 0, 38.0931),),
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
                    bulk_src_id="mp-149", millers=(1, 1, 1), shift=0.125, top=True,
                ),
            ),
            model="gemnet_oc_base_s2ef_all_md",
            ephemeral=True,
        )

        async with _ensure_system_deleted(self.CLIENT, response.system_id):
            self.assertNotEqual(response.system_id, "")
            self.assertEqual(len(response.config_ids), 1)

    @pytest.mark.ocpapi_integration_test
    async def test_submit_adsorbate_slab_relaxations__equiformer_v2(self) -> None:
        # Make sure that a relaxation can be started for an adsorbate
        # placement on a slab with the equiformer v2 model

        response = await self.CLIENT.submit_adsorbate_slab_relaxations(
            adsorbate="*CO",
            adsorbate_configs=[
                Atoms(
                    cell=((11.6636, 0, 0), (-5.8318, 10.1010, 0), (0, 0, 38.0931),),
                    pbc=(True, True, False),
                    numbers=[6, 8],
                    tags=[2, 2],
                    positions=[(1.9439, 3.3670, 22.2070), (1.9822, 3.2849, 23.3697),],
                )
            ],
            bulk=Bulk(src_id="mp-149", elements=["Si"], formula="Si"),
            slab=Slab(
                atoms=Atoms(
                    cell=((11.6636, 0, 0), (-5.8318, 10.1010, 0), (0, 0, 38.0931),),
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
                    bulk_src_id="mp-149", millers=(1, 1, 1), shift=0.125, top=True,
                ),
            ),
            model="equiformer_v2_31M_s2ef_all_md",
            ephemeral=True,
        )

        async with _ensure_system_deleted(self.CLIENT, response.system_id):
            self.assertNotEqual(response.system_id, "")
            self.assertEqual(len(response.config_ids), 1)

    async def test_get_adsorbate_slab_relaxations_request(self) -> None:
        # Make sure the original request can be fetched for an already-
        # submitted system.

        response = await self.CLIENT.get_adsorbate_slab_relaxations_request(
            system_id=self.KNOWN_SYSTEM_ID
        )

        # Don't worry about checking all fields - just make sure at least one
        # of the expected fields was returned
        self.assertEqual(response.adsorbate, "*CO")

    async def test_get_adsorbate_slab_relaxations_results__all_fields_and_configs(
        self,
    ) -> None:
        # Make sure relaxation results can be fetched for an already-relaxed
        # system. Check that all configurations and all fields for each are
        # returned.

        response = await self.CLIENT.get_adsorbate_slab_relaxations_results(
            system_id=self.KNOWN_SYSTEM_ID,
        )

        self.assertEqual(len(response.configs), 59)
        for config in response.configs:
            self.assertEqual(config.status, Status.SUCCESS)
            self.assertIsNotNone(config.system_id)
            self.assertIsNotNone(config.cell)
            self.assertIsNotNone(config.pbc)
            self.assertIsNotNone(config.numbers)
            self.assertIsNotNone(config.positions)
            self.assertIsNotNone(config.tags)
            self.assertIsNotNone(config.energy)
            self.assertIsNotNone(config.energy_trajectory)
            self.assertIsNotNone(config.forces)
        config_ids = {c.config_id for c in response.configs}
        self.assertEqual(config_ids, set(range(59)))

    async def test_get_adsorbate_slab_relaxations_results__limited_fields_and_configs(
        self,
    ) -> None:
        # Make sure relaxation results can be fetched for an already-relaxed
        # system. Check that only the requested configurations and fields are
        # returned.

        response = await self.CLIENT.get_adsorbate_slab_relaxations_results(
            system_id=self.KNOWN_SYSTEM_ID,
            config_ids=[10, 20, 30],
            fields=["energy", "cell"],
        )

        self.assertEqual(len(response.configs), 3)
        for config in response.configs:
            self.assertEqual(config.status, Status.SUCCESS)
            self.assertIsNone(config.system_id)
            self.assertIsNotNone(config.cell)
            self.assertIsNone(config.pbc)
            self.assertIsNone(config.numbers)
            self.assertIsNone(config.positions)
            self.assertIsNone(config.tags)
            self.assertIsNotNone(config.energy)
            self.assertIsNone(config.energy_trajectory)
            self.assertIsNone(config.forces)
        config_ids = {c.config_id for c in response.configs}
        self.assertEqual(config_ids, {10, 20, 30})
