import json
import os


def main(jsons, dft_ref_trajs, out_path):
    id2pos = {}
    ids = []
    for i in jsons:
        d = json.load(open(i, "r"))
        for j in d:
            ids.append(j[0])
            if j[0] in id2pos:
                print("repeat", j[0])
                pass
            else:
                id2pos[j[0]] = j[1]
    with open(dft_ref_trajs, "r") as f:
        randomids = f.read().splitlines()
    randomids = [os.path.split(i)[-1].split(".")[0] for i in randomids]
    n = len(id2pos)
    print("total", n)
    preds = []
    for i in range(n):
        preds.append(
            {
                "simulation_id": randomids[i],
                "positions": id2pos[i],
            }
        )
    print("Saving to", out_path)
    json.dump(preds, open(out_path, "w"))


if __name__ == "__main__":
    """
    If relaxations run in a multi-GPU configuration, merge
    relaxed_pos_[DEVICE #] json files into a single json file in the correct
    ordering necessary for evalAI.
    """

    jsons = [
        "PATH/TO/relaxed_pos_0.json",
        "PATH/TO/relaxed_pos_2.json",
        "PATH/TO/relaxed_pos_3.json",
    ]

    dft_ref_trajs = "PATH/TO/split_txt_file_with_ids"
    out_path = "relaxed_pos_full.json"
