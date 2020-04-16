import torch
import gc
import pickle
import numpy as np
import gpytorch

from .LBFGS import FullBatchLBFGS
from ..common.registry import registry
from ..models.gps import ExactGPModel


@registry.register_trainer("gpytorch")
class GPyTorchTrainer:
    ''' Much of this code was adapted from the GPyTorch tutorial documentation '''
    def __init__(self, convolution_trainer, train_x, train_y,
                 Gp=None, Optimizer=None, Likelihood=None, Loss=None,
                 lr=0.1, preconditioner_size=100,
                 device=None, n_devices=None):

        if Gp is None:
            Gp = ExactGPModel
        if Optimizer is None:
            Optimizer = FullBatchLBFGS
        if Likelihood is None:
            Likelihood = gpytorch.likelihoods.GaussianLikelihood
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if n_devices is None:
            n_devices = torch.cuda.device_count()
            print('Planning to run on {} GPUs.'.format(n_devices))
        if Loss is None:
            Loss = gpytorch.mlls.ExactMarginalLogLikelihood

        self.convolution_trainer = convolution_trainer
        self.train_x = train_x
        self.train_y = train_y
        self.device = device
        self.n_devices = n_devices
        self.preconditioner_size = preconditioner_size

        self.likelihood = Likelihood().to(self.device)
        self.gp = Gp(self.train_x, self.train_y, self.likelihood,
                     self.device, self.n_devices).to(self.device)
        self.optimizer = Optimizer(self.gp.parameters(), lr=lr)
        self.loss = Loss(self.likelihood, self.gp)

        self._calculate_checkpoint_size()

    def train(self, n_training_iter=20):
        self.gp.train()
        self.likelihood.train()

        with (gpytorch.beta_features.checkpoint_kernel(self.checkpoint_size),
              gpytorch.settings.max_preconditioner_size(self.preconditioner_size)):

            loss = self.__closure()
            loss.backward()

            for i in range(n_training_iter):
                options = {'closure': self.__closure,
                           'current_loss': loss,
                           'max_ls': 10}
                loss, _, _, _, _, _, _, fail = self.optimizer.step(options)

                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f'
                      % (i + 1, n_training_iter, loss.item(),
                         self.gp.covar_module.module.base_kernel.lengthscale.item(),
                         self.gp.likelihood.noise.item()))

                if fail:
                    print('Convergence reached!')
                    break

        print(f"Finished training on {self.train_x.size(0)} data points using {self.n_devices} GPUs.")
        return self.gp, self.likelihood

    def __closure(self):
        self.optimizer.zero_grad()
        output = self.gp(self.train_x)
        loss = -self.likelihood(output, self.train_y)
        return loss

    def _calculate_checkpoint_size(self):
        ''' Define routine for getting GPU settings '''
        N = self.train_x.size(0)

        # Find the optimum partition/checkpoint size by decreasing in powers of 2
        # Start with no partitioning (size = 0)
        settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

        for checkpoint_size in settings:
            print('Number of devices: {} -- Kernel partition size: {}'.format(self.n_devices, checkpoint_size))
            try:
                # Try a full forward and backward pass with this setting to check memory usage
                _, _ = self.train(n_training_iter=1)

                # when successful, break out of for-loop and jump to finally block
                break
            except RuntimeError as e:
                print('RuntimeError: {}'.format(e))
            except AttributeError as e:
                print('AttributeError: {}'.format(e))
            finally:
                # handle CUDA OOM error
                gc.collect()
                torch.cuda.empty_cache()
        self.checkpoint_size = checkpoint_size

    def predict(self, input_):
        # Make and save the predictions
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
            preds = self.gp(input_)
        targets_pred = preds.mean
        targets_std = preds.stddev.detach().cpu().numpy()
        return targets_pred, targets_std

    def save_gp(self, path='gp_state.pth'):
        torch.save(self.gp.state_dict(), (path))

    def load_gp(self, path):
        state_dict = torch.load(path)
        self.gp.load_state_dict(state_dict)
