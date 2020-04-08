from baselines.common.flags import flags
from baselines.trainers import BaseTrainer

if __name__ == "__main__":
    parser = flags.get_parser()
    args = parser.parse_args()

    trainer = BaseTrainer(args)
    trainer.load()

    trainer.train()
