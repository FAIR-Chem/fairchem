import os
import sys
import time

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports

if __name__ == "__main__":
    # Setup logging and add key arguments
    sys.argv.append("--mode=train")
    sys.argv.append("--config=configs/is2re/all/schnet/schnet.yml")
    sys.argv.append(
        "--checkpoint=checkpoints/2022-05-03-16-09-36-schnet/best_checkpoint.pt"
    )
    # sys.argv.append("--config=configs/is2re/all/schnet/schnet.yml")
    # sys.argv.append(
    #     "--checkpoint=checkpoints/2022-04-26-14-10-08-schnet/best_checkpoint.pt"
    # )

    start_time = time.time()

    # Load datasets
    metrics = {}
    for s in ["val_id", "val_ood_ads", "val_ood_cat", "val_ood_both"]:
        print("Validation set: ", s)

        # Load config
        parser = flags.get_parser()
        args, override_args = parser.parse_known_args()
        config = build_config(args, override_args)

        config["dataset"][-1] = {
            "src": "/network/projects/_groups/ocp/oc20/is2re/all/"
            + s
            + "/data.lmdb"
        }

        # Define trainer
        setup_imports()
        trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
            task=config["task"],
            model=config["model"],
            dataset=config["dataset"],
            optimizer=config["optim"],
            identifier=config["identifier"],
            timestamp_id=config.get("timestamp_id", None),
            run_dir=config.get("run_dir", "./"),
            is_debug=True,
            print_every=config.get("print_every", 100),
            seed=config.get("seed", 0),
            logger=config.get("logger", "tensorboard"),
            local_rank=config["local_rank"],
            amp=config.get("amp", False),
            cpu=config.get("cpu", False),
            slurm=config.get("slurm", {}),
            new_gnn=config.get("new_gnn", True),
        )

        # Load checkpoint
        checkpoint_path = os.path.join(
            os.path.dirname(config["checkpoint"]), "best_checkpoint.pt"
        )
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

        # Call validate function
        metric = trainer.validate(split="val", disable_tqdm=False)
        metrics[s] = metric

    # Print results
    print("Total time taken: ", time.time() - start_time)
    print(metric.keys())
    for k, v in metrics.items():
        store = []
        for key, val in v.items():
            store.append(round(val["metric"], 4))
        print(k, store)

    # Save results
    file = open("val_results.txt", "a+")
    file.write("\n")
    file.write("-----------------")
    file.write("\n")
    file.write("\n")
    file.writelines([str(args.config_yml) + "\n", args.checkpoint + "\n"])
    file.write(str(metric.keys()))
    file.write("\n")
    file.writelines(
        ["val_id ", " val_ood_ads ", " val_ood_cat ", " val_ood_both "]
    )
    file.write("\n")
    for k, v in metrics.items():
        store = []
        for key, val in v.items():
            store.append(round(val["metric"], 4))
        file.write(str(store))
        file.write("\n")
    file.close()
