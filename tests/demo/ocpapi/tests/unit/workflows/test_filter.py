from __future__ import annotations

import functools
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from io import StringIO
from typing import Any, List, Optional, Tuple
from unittest import IsolatedAsyncioTestCase, mock

from fairchem.demo.ocpapi.client import AdsorbateSlabConfigs, Atoms, Slab, SlabMetadata
from fairchem.demo.ocpapi.workflows import (
    keep_all_slabs,
    keep_slabs_with_miller_indices,
    prompt_for_slabs_to_keep,
)
from inquirer import prompt
from inquirer.events import KeyEventGenerator
from inquirer.render import ConsoleRender
from readchar import key


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

    async def test_prompt_for_slabs_to_keep(self) -> None:
        @dataclass
        class TestCase:
            message: str
            input: List[AdsorbateSlabConfigs]
            key_events: List[Any]
            expected: List[AdsorbateSlabConfigs]

        test_cases: List[TestCase] = [
            # If no adslabs are provided then none should be returned
            TestCase(
                message="no slabs provided",
                input=[],
                key_events=[],
                expected=[],
            ),
            # If adslabs are provided but none are selected then none
            # should be returned
            TestCase(
                message="no slabs selected",
                input=[
                    _new_adslab(miller_indices=(1, 0, 0)),
                    _new_adslab(miller_indices=(2, 0, 0)),
                    _new_adslab(miller_indices=(3, 0, 0)),
                ],
                key_events=[key.ENTER],
                expected=[],
            ),
            # If adslabs are provided and some are selected then those
            # should be returned
            TestCase(
                message="some slabs selected",
                input=[
                    _new_adslab(miller_indices=(1, 0, 0)),
                    _new_adslab(miller_indices=(2, 0, 0)),
                    _new_adslab(miller_indices=(3, 0, 0)),
                ],
                key_events=[
                    key.SPACE,  # Select first slab
                    key.DOWN,  # Move to second slab
                    key.DOWN,  # Move to third slab
                    key.SPACE,  # Select third slab
                    key.ENTER,  # Finish
                ],
                expected=[
                    _new_adslab(miller_indices=(1, 0, 0)),
                    _new_adslab(miller_indices=(3, 0, 0)),
                ],
            ),
        ]

        for case in test_cases:
            with ExitStack() as es:
                es.enter_context(self.subTest(msg=case.message))

                # prompt_for_slabs_to_keep() creates an interactive prompt
                # that the user can select from. Here we inject key presses
                # to simulate a user interacting with the prompt. First we
                # need to direct stdin and stdout to our own io objects.
                orig_stdin = sys.stdin
                orig_stdout = sys.stdout
                try:
                    sys.stdin = StringIO()
                    sys.stdout = StringIO()

                    # Now we create a inquirer.ConsoleRender instance that
                    # uses the key_events (key presses) in the current test
                    # case.
                    it = iter(case.key_events)
                    renderer = ConsoleRender(
                        event_generator=KeyEventGenerator(lambda: next(it))
                    )

                    # Now inject our renderer into the prompt
                    es.enter_context(
                        mock.patch(
                            "inquirer.prompt",
                            side_effect=functools.partial(
                                prompt,
                                render=renderer,
                            ),
                        )
                    )

                    # Finally run the filter
                    adslab_filter = prompt_for_slabs_to_keep()
                    actual = await adslab_filter(case.input)
                    self.assertEqual(case.expected, actual)

                finally:
                    sys.stdin = orig_stdin
                    sys.stdout = orig_stdout
