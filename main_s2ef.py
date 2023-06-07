import argparse
import logging
from pathlib import Path

from ll import Runner, Trainer

from ocpmodels.trainers.base import S2EFConfig, S2EFModule

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("config", type=Path, help="Path to config file")
    args = parser.parse_args()

    # Parse the config
    config = S2EFConfig.from_file(args.config)

    # Define the runner main function.
    # This is called locally when using runner.local(config)
    # or on every GPU when using runner.submit([config, ...]).
    def run(config: S2EFConfig):
        model = S2EFModule(config)
        trainer = Trainer(config)
        trainer.fit(model)

    # Create the runner and run locally
    runner = Runner(run)
    runner.local(config)


if __name__ == "__main__":
    main()
