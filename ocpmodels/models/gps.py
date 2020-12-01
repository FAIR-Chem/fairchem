import gpytorch

from ocpmodels.common.registry import registry


@registry.register_model("exact_gp")
class ExactGP(gpytorch.models.ExactGP):
    """ Modified from GPyTorch tutorial documentation """

    def __init__(
        self,
        MeanKernel,
        CovKernel,
        OutputDist,
        likelihood,
        train_x,
        train_y,
        output_device,
        n_devices,
    ):
        self.OutputDist = OutputDist

        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MeanKernel()
        base_covar_module = gpytorch.kernels.ScaleKernel(CovKernel())

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module,
            device_ids=range(n_devices),
            output_device=output_device,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.OutputDist(mean_x, covar_x)
