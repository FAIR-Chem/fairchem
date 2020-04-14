import os
import yaml
import datetime
import torch
from ocpmodels.common.registry import registry
from .base_trainer import BaseTrainer


@registry.register_trainer("sktrainer")
class SKTrainer(BaseTrainer):
    '''
    Extended version of `BaseTrainer` that wraps up some of the overhead for us
    and calls out some of the arguments explicitly. The flow should be:
        trainer = SKTrainer(*)
        trainer.load_task(*)
        trainer.load_model(*)
        trainer.load_criterion(*)   # optional
        trainer.load_optimizer(*)   # optional
        trainer.train(*)
        trainer.predict(*)          # optional
    '''
    def __init__(self, run_dir='.', is_debug=False, is_vis=False,
                 print_every=100, seed=None, logger=None):
        self.is_debug = is_debug
        self.is_vis = is_vis
        self.config['cmd'] = {'print_every': print_every, 'seed': seed, 'logger': logger}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_seed_from_config()
        self._create_subdirs(run_dir)

    def _create_subdirs(self, run_dir):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.cmd['checkpoint_dir'] = os.path.join(run_dir, "checkpoints", timestamp)
        self.cmd['results_dir'] = os.path.join(run_dir, "results", timestamp)
        self.cmd['logs_dir'] = os.path.join(run_dir, "logs", self.config["logger"], timestamp)
        os.makedirs(self.cmd['checkpoint_dir'])
        os.makedirs(self.cmd['results_dir'])
        os.makedirs(self.cmd['logs_dir'])

    def train(self, train_size, val_size, test_size):
        self.assert_config()
        self.config['train_size'] = train_size
        self.config['val_size'] = val_size
        self.config['test_size'] = test_size

        print(yaml.dump(self.config, default_flow_style=False))
        self.train()

    def assert_config(self):
        # required methods
        for attribute in ['task', 'model']:
            if attribute not in self.config:
                raise RuntimeError('The %s has not yet been loaded. Please call '
                                   'the load_%s method.' % (attribute, attribute))

        # optional methods that fall back to defaults
        if 'criterion' not in self:
            self.load_criterion()
        if 'optim' not in self.config:
            self.load_optimizer()

    def load_task(self, src, dataset, description, labels, metric, type_):
        self.config['dataset'] = {'src': src}
        self.config['task'] = {'dataset': dataset,
                               'description': description,
                               'labels': labels,
                               'metric': metric,
                               'type': type_}
        super(SKTrainer, self).load_task()

    def load_model(self, model):
        self.config['model'] = model
        super(SKTrainer, self).load_model()

    def load_optimizer(self, batch_size=64, lr_gamma=0.1, lr_initial=0.001,
                       lr_milestones=None, max_exochs=50, warmup_epochs=10,
                       warmup_factor=0.2):
        # default for mutable object
        if lr_milestones is None:
            lr_milestones = [100, 150]

        self.config['optim'] = {'batch_size': batch_size,
                                'lr_gamma': lr_gamma,
                                'lr_initial': lr_initial,
                                'lr_milestones': lr_milestones,
                                'max_exochs': max_exochs,
                                'warmup_epochs': warmup_epochs,
                                'warmup_factor': warmup_factor}
        super(SKTrainer, self).load_optimizer()

    def predict(self):
        raise NotImplementedError
