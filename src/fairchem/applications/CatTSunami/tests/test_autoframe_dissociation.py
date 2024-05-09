from ocpneb.core.autoframe import (
    AutoFrameDissociation,
    is_edge_list_respected,
    interpolate_and_correct_frames,
)
import pickle
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
from ocpneb.core import Reaction
from fairchem.data.oc.databases.pkls import ADSORBATES_PKL_PATH
from ocpneb.databases import DISSOCIATION_REACTION_DB_PATH

def get_ads_syms(adslab):
    adsorbate = adslab[[idx for idx, tag in enumerate(adslab.get_tags()) if tag == 2]]
    syms = adsorbate.get_chemical_symbols()
    syms_str = ""
    for sym in syms:
        syms_str += sym
    return syms_str


class TestAutoframe:
    def test_overall_functionality(self):
        with open("autoframe_inputs_dissociation.pkl", "rb") as f:
            inputs = pickle.load(f)

        num_frames = 10
        reactant_systems = inputs["reactant_systems"]
        product1_systems = inputs["product1_systems"]
        product2_systems = inputs["product2_systems"]

        reactant_energies = inputs["reactant_energies"]
        product1_energies = inputs["product1_energies"]
        product2_energies = inputs["product2_energies"]

        reactant_system = reactant_systems[
            reactant_energies.index(min(reactant_energies))
        ]
        reaction = Reaction(
            reaction_db_path=DISSOCIATION_REACTION_DB_PATH,
            reaction_id_from_db=8,
            adsorbate_db_path="/private/home/brookwander/Open-Catalyst-Dataset/ocdata/databases/pkls/adsorbates.pkl",
        )
        checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
        calc1 = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
        af = AutoFrameDissociation(
            reaction,
            reactant_system,
            product1_systems,
            product1_energies,
            product2_systems,
            product2_energies,
            2,
            3,
            1,
        )
        neb_frames_sets, map_idx_list = af.get_neb_frames(
            calc1,
            n_frames=num_frames,
            n_pdt1_sites=2,
            n_pdt2_sites=2,
            fmax=0.05,
        )
        neb_frames_len = [len(neb_set) == num_frames for neb_set in neb_frames_sets]
        syms_str_agree = [
            get_ads_syms(frame_set[0]) == get_ads_syms(frame_set[-1])
            for frame_set in neb_frames_sets
        ]

        new_frames = interpolate_and_correct_frames(
            neb_frames_sets[0][0],
            neb_frames_sets[0][-1],
            5,
            reaction,
            map_idx_list[0],
        )

        assert all(syms_str_agree)
        assert len(new_frames) == 5
        assert all(neb_frames_len)
        assert len(neb_frames_sets) == 2

    def test_reactant_connectedness_check(self):
        """
        If the input is modified, be sure to have examples where the
        reactant isnt connected and examples where it is but across a
        periodic boundary.
        """
        with open("autoframe_inputs_dissociation.pkl", "rb") as f:
            inputs = pickle.load(f)

        reactant_systems = inputs["reactant_systems"]
        reaction = Reaction(
            reaction_db_path=DISSOCIATION_REACTION_DB_PATH,
            reaction_id_from_db=8,
            adsorbate_db_path=ADSORBATES_PKL_PATH,
        )

        connected = 0
        unconnected = 0
        for reactant in reactant_systems:
            if is_edge_list_respected(reactant, reaction.edge_list_initial):
                connected += 1
            else:
                unconnected += 1
        print(connected, unconnected)
        assert connected == 178
        assert unconnected == 2
