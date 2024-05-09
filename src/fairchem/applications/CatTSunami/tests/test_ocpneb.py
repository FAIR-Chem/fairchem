import pickle


import numpy as np

from ase.neb import DyNEB
from ocpneb.core import OCPdyNEB
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS
from fairchem.core.models.model_registry import model_name_to_local_file



class TestNEB:
    def test_force_call(self):
        images = pickle.load(open("neb_frames.pkl", "rb"))
        checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
        batched = OCPdyNEB(
            images=images,
            checkpoint_path=checkpoint_path,
            cpu=True,
            batch_size=4,
            k=0.25,
        )

        for image in images[1:-1]:
            image.calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)
        unbatched = DyNEB(images, k=0.25)

        forces = batched.get_forces()
        energies = batched.get_potential_energy()

        forces_ub = unbatched.get_forces()
        energies_ub = unbatched.get_potential_energy()

        mismatch = np.isclose(forces, forces_ub, atol=1e-3).all(axis=1) == False

        assert np.isclose(energies, energies_ub, atol=1e-3).all()
        assert mismatch.sum() == 0

    def test_neb_call(self):
        images = pickle.load(open("neb_frames.pkl", "rb"))
        checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
        batched = OCPdyNEB(
            images=images,
            checkpoint_path=checkpoint_path,
            cpu=True,
            batch_size=4,
            k=0.25,
        )

        batched_opt = BFGS(batched, trajectory="batched.traj")
        batched_opt.run(fmax=0.05, steps=5)
        forces = batched.get_forces()
        energies = batched.get_potential_energy()

        images_ub = pickle.load(open("neb_frames.pkl", "rb"))
        for image in images_ub[1:-1]:
            image.calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)
        unbatched = DyNEB(images_ub, k=0.25)
        unbatched_opt = BFGS(unbatched, trajectory="unbatched.traj")
        unbatched_opt.run(fmax=0.05, steps=5)
        forces_ub = unbatched.get_forces()
        energies_ub = unbatched.get_potential_energy()

        mismatch = np.isclose(forces, forces_ub, atol=1e-3).all(axis=1) == False

        assert np.isclose(energies, energies_ub, atol=1e-3).all()
        assert mismatch.sum() == 0
