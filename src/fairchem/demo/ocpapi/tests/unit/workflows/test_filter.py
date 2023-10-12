from dataclasses import dataclass
from typing import List, Optional, Tuple
from unittest import IsolatedAsyncioTestCase

from ocpapi.client import AdsorbateSlabConfigs, Atoms, Slab, SlabMetadata
from ocpapi.workflows import keep_all_slabs, keep_slabs_with_miller_indices


# Function used to generate a new adslab instance. This filles the minimum
# set of required fields with default values. Inputs allow for overriding
# those defaults.
def _new_adslab(
    miller_indices: Optional[Tuple[int, int, int]] = None,
) -> AdsorbateSlabConfigs:
    return AdsorbateSlabConfigs(
        adsorbate_configs=[],
        slab=Slab(
            atoms=Atoms(
                cell=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                pbc=[True, True, False],
                numbers=[],
                positions=[],
                tags=[],
            ),
            metadata=SlabMetadata(
                bulk_src_id="bulk_id",
                millers=miller_indices or (2, 1, 0),
                shift=0.5,
                top=True,
            ),
        ),
    )


class TestFilter(IsolatedAsyncioTestCase):
    async def test_keep_all_slabs(self) -> None:
        @dataclass
        class TestCase:
            message: str
            input: List[AdsorbateSlabConfigs]
            expected: List[AdsorbateSlabConfigs]

        test_cases: List[TestCase] = [
            # If no adslabs are provided then none should be returned
            TestCase(
                message="empty list",
                input=[],
                expected=[],
            ),
            # If adslabs are provided, all should be returned
            TestCase(
                message="non-empty list",
                input=[
                    _new_adslab(),
                    _new_adslab(),
                ],
                expected=[
                    _new_adslab(),
                    _new_adslab(),
                ],
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                adslab_filter = keep_all_slabs()
                actual = await adslab_filter(case.input)
                self.assertEqual(case.expected, actual)

    async def test_keep_slabs_with_miller_indices(self) -> None:
        @dataclass
        class TestCase:
            message: str
            adslab_filter: keep_slabs_with_miller_indices
            input: List[AdsorbateSlabConfigs]
            expected: List[AdsorbateSlabConfigs]

        test_cases: List[TestCase] = [
            # If no miller indices are defined, then no slabs should be kept
            TestCase(
                message="no miller indices allowed",
                adslab_filter=keep_slabs_with_miller_indices(miller_indices=[]),
                input=[
                    _new_adslab(miller_indices=(1, 0, 0)),
                    _new_adslab(miller_indices=(1, 1, 0)),
                    _new_adslab(miller_indices=(1, 1, 1)),
                ],
                expected=[],
            ),
            # If no slabs are defined then nothing should be returned
            TestCase(
                message="no slabs provided",
                adslab_filter=keep_slabs_with_miller_indices(
                    miller_indices=[(1, 1, 1)]
                ),
                input=[],
                expected=[],
            ),
            # Any miller indices that do match should be kept
            TestCase(
                message="some miller indices matched",
                adslab_filter=keep_slabs_with_miller_indices(
                    miller_indices=[
                        (1, 0, 1),  # Won't match anything
                        (1, 0, 0),  # Will match
                        (1, 1, 1),  # Will match
                    ]
                ),
                input=[
                    _new_adslab(miller_indices=(1, 0, 0)),
                    _new_adslab(miller_indices=(1, 1, 0)),
                    _new_adslab(miller_indices=(1, 1, 1)),
                ],
                expected=[
                    _new_adslab(miller_indices=(1, 0, 0)),
                    _new_adslab(miller_indices=(1, 1, 1)),
                ],
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.message):
                actual = await case.adslab_filter(case.input)
                self.assertEqual(case.expected, actual)
