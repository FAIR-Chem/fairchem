from baselines.common.flags import flags
from baselines.common.registry import registry
from baselines.trainers import ActiveDiscoveryTrainer

if __name__ == "__main__":
    parser = flags.get_parser()
    args = parser.parse_args()

    trainer = ActiveDiscoveryTrainer(args)
    trainer.load()

    trainer.train()
