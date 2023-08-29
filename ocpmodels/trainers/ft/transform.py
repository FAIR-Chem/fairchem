from torch_geometric.data import Data

from ocpmodels.models.gemnet_oc_mt.goc_graph import Cutoffs, MaxNeighbors
from ocpmodels.trainers.mt.config import TransformConfigs
from ocpmodels.trainers.mt.transform import (
    _common_transform,
    _common_transform_all,
    _generate_graphs,
)


def qm9_transform(data: Data, *, config: TransformConfigs, training: bool):
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(8.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=False,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data
