import gc
import pickle

import gpytorch
import numpy as np
import torch

from ocpmodels.common.lbfgs import FullBatchLBFGS
from ocpmodels.common.registry import registry
from ocpmodels.models import ExactGP


@registry.register_trainer("gpytorch")
class GPyTorchTrainer:
    """ Much of this code was adapted from the GPyTorch tutorial documentation """

    def __init__(
        self,
        Gp=None,
        Optimizer=None,
        Likelihood=None,
        Loss=None,
        kernel=None,
        cov_matrix=None,
        lr=0.1,
        preconditioner_size=100,
        device=None,
        n_devices=None,
    ):

        if Gp is None:
            Gp = ExactGP
        if Optimizer is None:
            Optimizer = FullBatchLBFGS
        if Likelihood is None:
            Likelihood = gpytorch.likelihoods.GaussianLikelihood
        if Loss is None:
            Loss = gpytorch.mlls.ExactMarginalLogLikelihood
        if kernel is None:
            kernel = gpytorch.kernels.MaternKernel()
        if cov_matrix is None:
            cov_matrix = gpytorch.distributions.MultivariateNormal
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if n_devices is None:
            n_devices = torch.cuda.device_count()
            print("Planning to run on {} GPUs.".format(n_devices))

        self.device = device
        self.n_devices = n_devices
        self.preconditioner_size = preconditioner_size

        self.Gp = Gp
        self.Optimizer = Optimizer
        self.Loss = Loss
        self.likelihood = Likelihood().to(self.device)
        self.kernel = kernel
        self.cov_matrix = cov_matrix

    def train(self, train_x, train_y, lr=0.1, n_training_iter=20):
        self._init_gp(train_x, train_y, lr)
        self._calculate_checkpoint_size(train_x, train_y, lr)

        self._train(
            train_x=train_x,
            train_y=train_y,
            checkpoint_size=self.checkpoint_size,
            lr=lr,
            n_training_iter=n_training_iter,
        )

    def _init_gp(self, train_x, train_y, lr):
        self.gp = self.Gp(
            kernel=self.kernel,
            cov_matrix=self.cov_matrix,
            likelihood=self.likelihood,
            train_x=train_x,
            train_y=train_y,
            device=self.device,
            n_devices=self.n_devices
        ).to(self.device)
        self.loss = self.Loss(self.likelihood, self.gp)
        self.optimizer = self.Optimizer(self.gp.parameters(), lr=lr)

    def _calculate_checkpoint_size(self, train_x, train_y, lr):
        """ Runs through one set of training loops to figure out the best
        checkpointing size to use """
        N = train_x.size(0)

        # Find the optimum partition/checkpoint size by decreasing in powers of 2
        # Start with no partitioning (size = 0)
        settings = [0] + [
            int(n)
            for n in np.ceil(N / 2 ** np.arange(1, np.floor(np.log2(N))))
        ]

        for checkpoint_size in settings:
            print(
                "Number of devices: {} -- Kernel partition size: {}".format(
                    self.n_devices, checkpoint_size
                )
            )
            try:
                # Try a full forward and backward pass with this setting to check memory usage
                _, _ = self._train(
                    train_x, train_y, checkpoint_size, lr, n_training_iter=1
                )

                # when successful, break out of for-loop and jump to finally block
                break
            except RuntimeError as e:
                print("RuntimeError: {}".format(e))
            except AttributeError as e:
                print("AttributeError: {}".format(e))
            finally:
                # handle CUDA OOM error
                gc.collect()
                torch.cuda.empty_cache()
        self.checkpoint_size = checkpoint_size

    def _train(
        self, train_x, train_y, checkpoint_size, lr=0.1, n_training_iter=20
    ):
        self.gp.train()
        self.likelihood.train()

        with gpytorch.beta_features.checkpoint_kernel(
            checkpoint_size
        ), gpytorch.settings.max_preconditioner_size(self.preconditioner_size):

            def closure():
                self.optimizer.zero_grad()
                output = self.gp(train_x)
                loss = -self.loss(output, train_y)
                return loss

            loss = closure()
            loss.backward()

            for i in range(n_training_iter):
                options = {
                    "closure": closure,
                    "current_loss": loss,
                    "max_ls": 10,
                }
                loss, _, _, _, _, _, _, fail = self.optimizer.step(options)

                print(
                    "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                    % (
                        i + 1,
                        n_training_iter,
                        loss.item(),
                        self.gp.covar_module.module.base_kernel.lengthscale.item(),
                        self.gp.likelihood.noise.item(),
                    )
                )

                if fail:
                    print("Convergence reached!")
                    break

        print(
            f"Finished training on {train_x.size(0)} data points using {self.n_devices} GPUs."
        )
        return self.gp, self.likelihood

    def predict(self, input_):
        # Make and save the predictions
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(
            1000
        ):
            preds = self.gp.forward(input_)
        targets_pred = preds.mean
        targets_std = preds.stddev
        return targets_pred, targets_std

    def save_state(self, path='gp_state.pth'):
        torch.save(self.gp.state_dict(), (path))

    def load_state(self, path):
        state_dict = torch.load(path)
        self.gp.load_state_dict(state_dict)
