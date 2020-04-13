import argparse
import sys


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Graph Networks for Electrocatalyst Design"
        )
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        self.parser.add_argument(
            "--config-yml",
            default="configs/ulissigroup_co/cgcnn.yml",
            help="Path to a config file listing data, model, optim parameters.",
        )
        self.parser.add_argument(
            "--config-override",
            default=None,
            help="Optional override for parameters defined in config yaml",
        )
        self.parser.add_argument(
            "--identifier",
            default="",
            help="Experiment identifier to append to checkpoint/log/result directory",
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether this is a debugging run or not",
        )
        self.parser.add_argument(
            "--vis",
            action="store_true",
            help="Whether to visualize a few extra things",
        )
        self.parser.add_argument(
            "--num-workers",
            default=0,
            type=int,
            help="Number of dataloader workers (default: 0 i.e. use main proc)",
        )
        self.parser.add_argument(
            "--print-every",
            default=10,
            type=int,
            help="Log every N iterations (default: 10)",
        )
        self.parser.add_argument(
            "--seed", default=0, type=int, help="Seed for torch, cuda, numpy"
        )


flags = Flags()
