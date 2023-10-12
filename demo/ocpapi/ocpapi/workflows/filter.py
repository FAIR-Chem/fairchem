from typing import Iterable, List, Set, Tuple

from ocpapi.client import AdsorbateSlabConfigs


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
