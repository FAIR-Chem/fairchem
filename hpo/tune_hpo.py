import os
import torch
import torch.optim as optim
import torch.nn as nn

import ray
from ray import tune
# from ray.tune.schedulers import ASHAScheduler

import sys
sys.path.append("/global/homes/b/bwood/machine_learning/ulissi_cnn/hpo/ocp_cgcnn/cgcnn")

from baselines.modules.normalizer import Normalizer
from baselines.models.cgcnn import CGCNN
from baselines.datasets.base import BaseDataset
from train_hpo import train, validate


class TrainCGCNN(tune.Trainable):
    # define the device and the model
    def _setup(self, config):
        use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dataset = BaseDataset(config.get("data_config"))
        self.train_loader, self.val_loader, self.test_loader = \
            self.dataset.get_dataloaders(batch_size=config.get("batch_size", 80))
        self.model = CGCNN(num_atoms=self.train_loader.dataset.data.x.shape[-1],
                           bond_feat_dim=self.train_loader.dataset.data.edge_attr.shape[-1],
                           num_targets=1,
                           atom_embedding_size=config.get("atom_embedding_size", 64)).to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.get("lr", 0.01))
        self.normalizer = Normalizer(self.train_loader.dataset.data.y, self.device)

    def _train(self):
        t_loss, t_mae = train(self.model, self.criterion, self.optimizer,
                              self.train_loader, self.normalizer, self.device)
        v_loss, v_mae = validate(self.model, self.criterion, self.optimizer,
                                 self.val_loader, self.normalizer, self.device)
        return {"training_loss": t_loss, "training_mae": t_mae, "validation_loss": v_loss, "validation_mae": v_mae}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
