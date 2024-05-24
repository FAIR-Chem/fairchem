from typing import Iterable, List, Set, Tuple

from fairchem.demo.ocpapi.client import AdsorbateSlabConfigs, SlabMetadata


class keep_all_slabs:
    """
    Adslab filter than returns all slabs.
    """

    async def __call__(
        self,
        adslabs: List[AdsorbateSlabConfigs],
    ) -> List[AdsorbateSlabConfigs]:
        return adslabs


class keep_slabs_with_miller_indices:
    """
    Adslab filter that keeps any slabs with the configured miller indices.
    Slabs with other miller indices will be ignored.
    """

    def __init__(self, miller_indices: Iterable[Tuple[int, int, int]]) -> None:
        """
        Args:
            miller_indices: The list of miller indices that will be allowed.
                Slabs with any other miller indices will be dropped by this
                filter.
        """
        self._unique_millers: Set[Tuple[int, int, int]] = set(miller_indices)

    async def __call__(
        self,
        adslabs: List[AdsorbateSlabConfigs],
    ) -> List[AdsorbateSlabConfigs]:
        return [
            adslab
            for adslab in adslabs
            if adslab.slab.metadata.millers in self._unique_millers
        ]


class prompt_for_slabs_to_keep:
    """
    Adslab filter than presents the user with an interactive prompt to choose
    which of the input slabs to keep.
    """

    @staticmethod
    def _sort_key(
        adslab: AdsorbateSlabConfigs,
    ) -> Tuple[Tuple[int, int, int], float, str]:
        """
        Generates a sort key from the input adslab. Returns the miller indices,
        shift, and top/bottom label so that they will be sorted by those values
        in that order.
        """
        metadata: SlabMetadata = adslab.slab.metadata
        return (metadata.millers, metadata.shift, metadata.top)

    async def __call__(
        self,
        adslabs: List[AdsorbateSlabConfigs],
    ) -> List[AdsorbateSlabConfigs]:
        from inquirer import Checkbox, prompt

        # Break early if no adslabs were provided
        if not adslabs:
            return adslabs

        # Sort the input list so the options are grouped in a sensible way
        adslabs = sorted(adslabs, key=self._sort_key)

        # List of options to present to the user. The first item in each tuple
        # will be presented to the user in the prompt. The second item in each
        # tuple (indices from the input list of adslabs) will be returned from
        # the prompt.
        choices: List[Tuple[str, int]] = [
            (
                (
                    f"{adslab.slab.metadata.millers} "
                    f"{'top' if adslab.slab.metadata.top else 'bottom'} "
                    "surface shifted by "
                    f"{round(adslab.slab.metadata.shift, 3)}; "
                    f"{len(adslab.adsorbate_configs)} unique adsorbate "
                    "placements to relax"
                ),
                idx,
            )
            for idx, adslab in enumerate(adslabs)
        ]
        checkbox: Checkbox = Checkbox(
            "adslabs",
            message=(
                "Choose surfaces to relax (up/down arrows to move, "
                "space to select, enter when finished)"
            ),
            choices=choices,
        )
        selected_indices: List[int] = prompt([checkbox])["adslabs"]

        # Return the adslabs that were chosen
        return [adslabs[i] for i in selected_indices]
