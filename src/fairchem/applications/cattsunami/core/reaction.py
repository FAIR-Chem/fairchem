import pickle
import random


class Reaction:
    """
    Initialize Reaction object
    """

    def __init__(
        self,
        reaction_db_path: str,
        adsorbate_db_path: str,
        reaction_id_from_db: int = None,
        reaction_str_from_db: str = None,
        reaction_type: str = None,
    ):
        self.reaction_db_path = reaction_db_path
        reaction_db = pickle.load(open(reaction_db_path, "rb"))
        adsorbate_db = pickle.load(open(adsorbate_db_path, "rb"))

        if reaction_id_from_db is not None:
            self.reaction_id_from_db = reaction_id_from_db
        elif reaction_str_from_db is not None:
            self.reaction_id_from_db = [
                idx
                for idx, reaction in enumerate(reaction_db)
                if reaction["reaction"] == reaction_str_from_db
            ][0]
        elif reaction_type is not None:
            viable_reactions = [
                idx
                for idx, entry in enumerate(reaction_db)
                if entry["reaction_type"] == reaction_type
            ]
            self.reaction_id_from_db = random.choice(viable_reactions)

        else:
            self.reaction_id_from_db = random.choice(range(len(reaction_db)))

        entry = reaction_db[self.reaction_id_from_db]
        if entry["reaction_type"] == "dissociation":
            self.reaction_type = "dissociation"
            self.reaction_str_from_db = entry["reaction"]
            self.reactant1_idx = entry["reactant"]
            self.product1_idx = entry["product1"]
            self.product2_idx = entry["product2"]
            self.idx_mapping = entry["idx_mapping"]
            self.edge_list_initial = entry["edge_indices_initial"]
            self.edge_list_final = entry["edge_indices_final"]
            self.reactant1 = adsorbate_db[self.reactant1_idx][0]
            self.binding_atom_idx_reactant1 = adsorbate_db[self.reactant1_idx][2][0]
            self.product1 = adsorbate_db[self.product1_idx][0]
            self.binding_atom_idx_product1 = adsorbate_db[self.product1_idx][2][0]
            self.product2 = adsorbate_db[self.product2_idx][0]
            self.binding_atom_idx_product2 = adsorbate_db[self.product2_idx][2][0]

        elif entry["reaction_type"] == "desorption":
            self.reaction_type = "desorption"
            self.reaction_str_from_db = entry["reaction"]
            self.reactant1_idx = entry["reactant"]
            self.product1_idx = entry["product"]
            self.edge_list_initial = entry["edge_indices"]
            self.edge_list_final = entry["edge_indices"]
            self.reactant1 = adsorbate_db[self.reactant1_idx][0]
            self.idx_mapping = self.get_desorption_mapping(self.reactant1)
            self.binding_atom_idx_reactant1 = adsorbate_db[self.reactant1_idx][2][0]
            self.product1 = adsorbate_db[self.product1_idx][0]
            self.binding_atom_idx_product1 = adsorbate_db[self.product1_idx][2][0]

        elif entry["reaction_type"] == "transfer":
            self.reaction_type = "transfer"
            self.reaction_str_from_db = entry["reaction"]
            self.reactant1_idx = entry["reactant1"]
            self.reactant2_idx = entry["reactant2"]
            self.product1_idx = entry["product1"]
            self.product2_idx = entry["product2"]
            self.idx_mapping = entry["idx_mapping"]
            self.edge_list_initial = entry["edge_indices_initial"]
            self.edge_list_final = entry["edge_indices_final"]
            self.reactant1 = adsorbate_db[self.reactant1_idx][0]
            self.binding_atom_idx_reactant1 = adsorbate_db[self.reactant1_idx][2][0]
            self.reactant2 = adsorbate_db[self.reactant2_idx][0]
            self.binding_atom_idx_reactant2 = adsorbate_db[self.reactant2_idx][2][0]
            self.product1 = adsorbate_db[self.product1_idx][0]
            self.binding_atom_idx_product1 = adsorbate_db[self.product1_idx][2][0]
            self.product2 = adsorbate_db[self.product2_idx][0]
            self.binding_atom_idx_product2 = adsorbate_db[self.product2_idx][2][0]

    def get_desorption_mapping(self, reactant):
        """
        Get mapping for desorption reaction
        """
        mapping = {}
        for idx, atom in enumerate(reactant):
            mapping[idx] = idx
        return [mapping]
