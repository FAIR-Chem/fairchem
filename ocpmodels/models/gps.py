import gpytorch
from ocpmodels.common.registry import registry


@registry.register_model("exact_gp")
class ExactGP(gpytorch.models.ExactGP):
    """ Modified from GPyTorch tutorial documentation """

    def __init__(self, train_x, train_y, likelihood, device, n_devices,
                 kernel=None, cov_matrix=None):
        if kernel is None:
            kernel = gpytorch.kernels.MaternKernel()
        if cov_matrix is None:
            self.cov_matrix = gpytorch.distributions.MultivariateNormal

        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(kernel)

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices), device=device
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.cov_matrix(mean_x, covar_x)
