import time
from typing import Any, List
from unittest import IsolatedAsyncioTestCase, mock

from ocpapi.client import Atoms, Client, Model, Status
from ocpapi.workflows import (
    Lifetime,
    find_adsorbate_binding_sites,
    get_adsorbate_slab_relaxation_results,
    keep_slabs_with_miller_indices,
    wait_for_adsorbate_slab_relaxations,
)
from ocpapi.workflows.adsorbates import _get_absorbate_configs_on_slab


class TestAdsorbates(IsolatedAsyncioTestCase):
    """
    Tests that workflow methods run against a real server execute correctly.
    """

    TEST_HOST = "https://open-catalyst-api.metademolab.com/ocp"
    KNOWN_SYSTEM_ID = "f9eacd8f-748c-41dd-ae43-f263dd36d735"

    async def test_get_adsorbate_slab_relaxation_results(self) -> None:
        # The server is expected to omit some results when too many are
        # requested. Check that all results are fetched since test method
        # under test should retry until all results have been retrieved.

        # The system under test has 59 configs:
        # https://open-catalyst.metademolab.com/results/f9eacd8f-748c-41dd-ae43-f263dd36d735
        num_configs = 59

        client = Client(self.TEST_HOST)
        results = await get_adsorbate_slab_relaxation_results(
            system_id=self.KNOWN_SYSTEM_ID,
            config_ids=list(range(num_configs)),
            # Fetch a subset of fields to avoid transferring significantly more
            # data than we really need in this test
            fields=["energy", "pbc"],
            client=client,
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

        client = Client(self.TEST_HOST)
        await wait_for_adsorbate_slab_relaxations(
            system_id=self.KNOWN_SYSTEM_ID,
            check_immediately=False,
            slow_interval_sec=1,
            fast_interval_sec=1,
            client=client,
        )

        took = time.monotonic() - start
        self.assertGreaterEqual(took, 1)
        # Give a pretty generous upper bound so that this test is not flaky
        # when there is a poor connection or the server is busy
        self.assertLess(took, 5)

    async def test_find_adsorbate_binding_sites(self) -> None:
        # Run an end-to-end test to find adsorbate binding sites on the
        # surface of a bulk material.

        # By default, we'll end up running relaxations for dozens of adsorbate
        # placements on the bulk surface. This function selects out only the
        # first adsorbate configuration. It is injected into the
        # find_adsorbate_binding_sites workflow using the mock below. This
        # lets us run a smaller number of relaxations since we really don't
        # need to run dozens just to know that the method under test works.
        async def _get_first_absorbate_config_on_slab(
            *args: Any, **kwargs: Any
        ) -> List[Atoms]:
            all = await _get_absorbate_configs_on_slab(*args, **kwargs)
            return all[:1]

        with mock.patch(
            "ocpapi.workflows.adsorbates._get_absorbate_configs_on_slab",
            wraps=_get_first_absorbate_config_on_slab,
        ):
            client = Client(self.TEST_HOST)
            results = await find_adsorbate_binding_sites(
                adsorbate="*O",
                bulk="mp-30",
                model=Model.GEMNET_OC_BASE_S2EF_ALL_MD,
                slab_filter=keep_slabs_with_miller_indices([(1, 1, 1)]),
                client=client,
                # Since this is a test, delete the relaxations from the server
                # once results have been fetched.
                lifetime=Lifetime.DELETE,
            )

            self.assertEqual(1, len(results.slabs))
            self.assertEqual(1, len(results.slabs[0].configs))
            self.assertEqual(Status.SUCCESS, results.slabs[0].configs[0].status)
