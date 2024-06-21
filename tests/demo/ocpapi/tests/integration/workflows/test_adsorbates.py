import time
from typing import List
from unittest import IsolatedAsyncioTestCase

import pytest
import requests

from fairchem.demo.ocpapi.client import AdsorbateSlabConfigs, Client, Status
from fairchem.demo.ocpapi.workflows import (
    Lifetime,
    find_adsorbate_binding_sites,
    get_adsorbate_slab_relaxation_results,
    wait_for_adsorbate_slab_relaxations,
)


class TestAdsorbates(IsolatedAsyncioTestCase):
    """
    Tests that workflow methods run against a real server execute correctly.
    """

    CLIENT: Client = Client(
        host="open-catalyst-api.metademolab.com",
        scheme="https",
    )
    KNOWN_SYSTEM_ID: str = "f9eacd8f-748c-41dd-ae43-f263dd36d735"

    async def test_get_adsorbate_slab_relaxation_results(self) -> None:
        # The server is expected to omit some results when too many are
        # requested. Check that all results are fetched since test method
        # under test should retry until all results have been retrieved.

        # The system under test has 59 configs:
        # https://open-catalyst.metademolab.com/results/f9eacd8f-748c-41dd-ae43-f263dd36d735
        num_configs = 59

        results = await get_adsorbate_slab_relaxation_results(
            system_id=self.KNOWN_SYSTEM_ID,
            config_ids=list(range(num_configs)),
            # Fetch a subset of fields to avoid transferring significantly more
            # data than we really need in this test
            fields=["energy", "pbc"],
            client=self.CLIENT,
        )

        self.assertEqual(
            [r.status for r in results],
            [Status.SUCCESS] * num_configs,
        )

    async def test_wait_for_adsorbate_slab_relaxations(self) -> None:
        # This test runs against an already-finished set of relaxations.
        # The goal is not to check that the method waits when relaxations
        # are still running (that is covered in unit tests), but just to
        # ensure that the call to the API is made correctly and that the
        # function returns ~immediately because the relaxations are done.

        start = time.monotonic()

        await wait_for_adsorbate_slab_relaxations(
            system_id=self.KNOWN_SYSTEM_ID,
            check_immediately=False,
            slow_interval_sec=1,
            fast_interval_sec=1,
            client=self.CLIENT,
        )

        took = time.monotonic() - start
        self.assertGreaterEqual(took, 1)
        # Give a pretty generous upper bound so that this test is not flaky
        # when there is a poor connection or the server is busy
        self.assertLess(took, 5)

    @pytest.mark.ocpapi_integration_test
    async def test_find_adsorbate_binding_sites(self) -> None:
        # Run an end-to-end test to find adsorbate binding sites on the
        # surface of a bulk material.

        # By default, we'll end up running relaxations for dozens of adsorbate
        # placements on the bulk surface. This function selects out only the
        # first adsorbate configuration. This lets us run a smaller number of
        # relaxations since we really don't need to run dozens just to know
        # that the method under test works.
        async def _keep_first_adslab(
            adslabs: List[AdsorbateSlabConfigs],
        ) -> List[AdsorbateSlabConfigs]:
            return [
                AdsorbateSlabConfigs(
                    adsorbate_configs=adslabs[0].adsorbate_configs[:1],
                    slab=adslabs[0].slab,
                )
            ]

        results = await find_adsorbate_binding_sites(
            adsorbate="*O",
            bulk="mp-30",
            model="gemnet_oc_base_s2ef_all_md",
            adslab_filter=_keep_first_adslab,
            client=self.CLIENT,
            # Since this is a test, delete the relaxations from the server
            # once results have been fetched.
            lifetime=Lifetime.DELETE,
        )

        self.assertEqual(1, len(results.slabs))
        self.assertEqual(1, len(results.slabs[0].configs))
        self.assertEqual(Status.SUCCESS, results.slabs[0].configs[0].status)

        # Make sure that the adslabs being used have tags for sub-surface,
        # surface, and adsorbate atoms. Then make sure that forces are
        # exactly zero only for the sub-surface atoms.
        config = results.slabs[0].configs[0]
        self.assertEqual(
            {0, 1, 2},
            set(config.tags),
            "Expected tags for surface, sub-surface, and adsorbate atoms",
        )
        for tag, forces in zip(config.tags, config.forces):
            if tag == 0:  # Sub-surface atoms are fixed / have 0 forces
                self.assertEqual(forces, (0, 0, 0))
            else:
                self.assertNotEqual(forces, (0, 0, 0))

        # Make sure the UI URL is reachable
        response = requests.head(results.slabs[0].ui_url)
        self.assertEqual(200, response.status_code)
