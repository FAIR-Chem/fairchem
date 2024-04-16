from ase.io import read
from ocpneb.core.OCPdyNEB import OCPdyNEB
from ase.optimize import BFGS
import torch
import argparse
import pickle
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import os
from ase.vibrations import Vibrations
from vasp_interactive import VaspInteractive
from ocdata.utils.vasp import calculate_surface_k_points


def parse_neb_info(neb_frames, calc, conv, neb_id):
    entry = {}
    e_along_traj = []
    for frame in neb_frames:
        frame.calc = calc
        e_along_traj.append(frame.get_potential_energy())
    barrier_height = max(e_along_traj) - e_along_traj[0]
    E_rxn = e_along_traj[-1] - e_along_traj[0]
    if barrier_height <= 0.1 or barrier_height <= E_rxn + 0.1:
        barrierless = True
        ts_idx = None
    else:
        barrierless = False
        ts_idx = e_along_traj.index(max(e_along_traj))
    entry = {
        "neb_id": neb_id,
        "barrier_height": barrier_height,
        "E_rxn": E_rxn,
        "converged": conv,
        "barrierless": barrierless,
        "transition_state_idx": ts_idx,
    }
    return entry


def get_single_point(atoms, vasp_dir, vasp_flags):
    with VaspInteractive(
        directory=vasp_dir,
        **vasp_flags,
    ) as calc:
        atoms.calc = calc
        try:
            e_vasp = atoms.get_potential_energy()
            f_vasp = atoms.get_forces()
        except:
            print(
                "Initial calculation terminated in error. Unfortunately, this is a normal occurance despite the DFT calculation running fine."
            )
            atoms = read(f"{vasp_dir}/vasprun.xml")
            e_vasp = atoms.get_potential_energy()
            f_vasp = atoms.get_forces()
    return e_vasp, f_vasp


def get_vibrations(atoms, vasp_dir, vasp_flags, vasp_command):
    os.chdir(vasp_dir)
    try:
        with VaspInteractive(
            directory=vasp_dir, command=vasp_command, **vasp_flags
        ) as calc:
            atoms.calc = calc
            indices_to_vibrate = [
                idx for idx, tag in enumerate(atoms.get_tags()) if tag == 2
            ]

            vib = Vibrations(atoms, indices=indices_to_vibrate)
            vib.run()
            vib_frequencies = vib.get_frequencies()
    except:
        print(
            "Initial calculation terminated in error. This is common. Ill try changing ISYM."
        )
        vasp_flags["isym"] = -1
        try:
            with VaspInteractive(
                directory=vasp_dir, command=vasp_command, **vasp_flags
            ) as calc:
                atoms.calc = calc

                vib = Vibrations(atoms, indices=indices_to_vibrate)
                vib.run()
                vib_frequencies = vib.get_frequencies()
        except:
            print("Calculation still terminated in error. Returning empty list.")
            vib_frequencies = []
    return vib_frequencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--k")
    parser.add_argument("--n_frames")
    parser.add_argument("--fmax")
    parser.add_argument("--output_file_path")
    parser.add_argument("--batch_size")
    parser.add_argument("--delta_fmax_climb")
    parser.add_argument("--get_init_fin_sps")
    parser.add_argument("--mapping_file_path")
    parser.add_argument("--mapping_idx")
    parser.add_argument("--vasp_command")
    parser.add_argument("--vibration_calculation", default="False")

    args = parser.parse_args()

    with open(args.mapping_file_path, "rb") as f:
        mapping = pickle.load(f)

    file = mapping[int(args.mapping_idx)]
    checkpoint_path = args.checkpoint_path
    delta_fmax_climb = float(args.delta_fmax_climb)
    k = float(args.k)
    fmax = float(args.fmax)
    calc = OCPCalculator(checkpoint=checkpoint_path, cpu=False)
    model_id = checkpoint_path.split("/")[-1].split(".")[0]
    vasp_command = args.vasp_command

    os.makedirs(f"{args.output_file_path}/{model_id}", exist_ok=True)
    os.makedirs(f"{args.output_file_path}/{model_id}/vasp_files", exist_ok=True)

    try:
        neb_id = file.split("/")[-1].split(".")[0]
        os.makedirs(
            f"{args.output_file_path}/{model_id}/vasp_files/{neb_id}", exist_ok=True
        )
        vasp_dir = f"{args.output_file_path}/{model_id}/vasp_files/{neb_id}"
        traj = read(file, ":")
        neb_frames = traj[0 : int(args.n_frames)]
        conv = False
        torch.cuda.empty_cache()
        # Optimize:
        neb = OCPdyNEB(
            neb_frames,
            checkpoint_path=checkpoint_path,
            k=k,
            batch_size=int(args.batch_size),
        )
        optimizer = BFGS(
            neb,
            trajectory=f"{args.output_file_path}/{model_id}/{neb_id}-k_now.traj",
        )
        conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
        if conv:
            neb.climb = True
            conv = optimizer.run(fmax=fmax, steps=300)

        VASP_FLAGS = {
            "ibrion": -1,
            "nsw": 300,
            "isif": 0,
            "isym": 0,
            "lreal": "Auto",
            "ediffg": 0.0,
            "symprec": 1e-10,
            "encut": 350.0,
            "ncore": 1,
            "lcharg": False,
            "lwave": False,
            "gga": "RP",
            "pp": "PBE",
            "xc": "PBE",
            "kpts": calculate_surface_k_points(neb_frames[0]),
        }
        entry = parse_neb_info(neb_frames, calc, conv, neb_id)

        if args.get_init_fin_sps == "True":
            os.mkdir(f"{vasp_dir}/initial")
            os.mkdir(f"{vasp_dir}/final")

            e_init, f_init = get_single_point(
                neb_frames[0], f"{vasp_dir}/initial", VASP_FLAGS
            )
            e_fin, f_fin = get_single_point(
                neb_frames[-1], f"{vasp_dir}/final", VASP_FLAGS
            )
            entry["E_initial_SP"] = e_init
            entry["E_final_SP"] = e_fin
            entry["F_initial_SP"] = f_init
            entry["F_final_SP"] = f_fin
            entry["initial_atoms"] = neb_frames[0]
            entry["final_atoms"] = neb_frames[-1]
        print(entry["transition_state_idx"])
        if entry["transition_state_idx"] is not None:
            os.mkdir(f"{vasp_dir}/ts")
            e_ts, f_ts = get_single_point(
                neb_frames[entry["transition_state_idx"]], f"{vasp_dir}/ts", VASP_FLAGS
            )
            entry["E_TS_SP"] = e_ts
            entry["F_TS_SP"] = f_ts
            entry["transition_state_atoms"] = neb_frames[entry["transition_state_idx"]]
            if args.vibration_calculation == "True":
                vib_dir = f"{vasp_dir}/vib_calc"
                os.makedirs(vib_dir, exist_ok=True)
                vib_frequencies = get_vibrations(
                    neb_frames[entry["transition_state_idx"]],
                    vib_dir,
                    VASP_FLAGS,
                    vasp_command,
                )
                entry["vibrational_frequencies"] = vib_frequencies
        else:
            entry["E_TS_SP"] = None
            if args.vibration_calculation == "True":
                vib_frequencies = []
                entry["vibrational_frequencies"] = vib_frequencies

        with open(f"{args.output_file_path}/{model_id}/{neb_id}.pkl", "wb") as f:
            pickle.dump(entry, f)

    except:
        with open(f"{args.output_file_path}/{model_id}/{neb_id}.pkl", "wb") as f:
            pickle.dump(entry, f)
