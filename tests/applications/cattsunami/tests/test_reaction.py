from fairchem.applications.cattsunami.core import Reaction
import ase
from fairchem.applications.cattsunami.databases import DISSOCIATION_REACTION_DB_PATH
from fairchem.data.oc.databases.pkls import ADSORBATE_PKL_PATH


class TestReaction:
    def test_loading_from_id(self):
        reaction = Reaction(
            reaction_db_path=DISSOCIATION_REACTION_DB_PATH,
            reaction_id_from_db=0,
            adsorbate_db_path=ADSORBATE_PKL_PATH,
        )
        assert reaction.reaction_str_from_db == "*OH -> *O + *H"
        assert reaction.reactant1_idx == 2
        assert reaction.product1_idx == 0
        assert reaction.product2_idx == 1
        assert reaction.idx_mapping == [{0: 0, 1: 1}]
        assert reaction.edge_list_initial[0] == (0, 1)

    def test_loading_from_str(self):
        reaction = Reaction(
            reaction_db_path=DISSOCIATION_REACTION_DB_PATH,
            adsorbate_db_path=ADSORBATE_PKL_PATH,
            reaction_str_from_db="*OH -> *O + *H",
        )
        assert reaction.reaction_str_from_db == "*OH -> *O + *H"
        assert reaction.reactant1_idx == 2
        assert reaction.product1_idx == 0
        assert reaction.product2_idx == 1
        assert reaction.idx_mapping == [{0: 0, 1: 1}]
        assert reaction.edge_list_initial[0] == (0, 1)

    def test_loading_from_random(self):
        reaction = Reaction(
            reaction_db_path=DISSOCIATION_REACTION_DB_PATH,
            adsorbate_db_path=ADSORBATE_PKL_PATH,
        )
        assert len(reaction.idx_mapping[0]) == len(reaction.reactant1)
        assert type(reaction.product1) == ase.Atoms
        assert type(reaction.product2) == ase.Atoms
        assert len(reaction.edge_list_initial) == len(reaction.edge_list_final) + 1
