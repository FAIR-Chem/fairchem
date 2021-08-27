"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import torch

from ..utils import read_value_json, update_json


class AutomaticFit:
    """
    All added variables are processed in the order of creation.
    """

    activeVar = None
    queue = None
    fitting_mode = False

    def __init__(self, variable, scale_file, name):
        self.variable = variable  # variable to find value for
        self.scale_file = scale_file
        self._name = name

        self._fitted = False
        self.load_maybe()

        # first instance created
        if AutomaticFit.fitting_mode and not self._fitted:

            # if first layer set to active
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []  # initialize
            # else add to queue
            else:
                self._add2queue()  # adding variables to list fill fail in graph mode

    def reset():
        AutomaticFit.activeVar = None
        AutomaticFit.all_processed = False

    def fitting_completed():
        return AutomaticFit.queue is None

    def set2fitmode():
        AutomaticFit.reset()
        AutomaticFit.fitting_mode = True

    def _add2queue(self):
        logging.debug(f"Add {self._name} to queue.")
        # check that same variable is not added twice
        for var in AutomaticFit.queue:
            if self._name == var._name:
                raise ValueError(
                    f"Variable with the same name ({self._name}) was already added to queue!"
                )
        AutomaticFit.queue += [self]

    def set_next_active(self):
        """
        Set the next variable in the queue that should be fitted.
        """
        queue = AutomaticFit.queue
        if len(queue) == 0:
            logging.debug("Processed all variables.")
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None  # reset to None
            return
        AutomaticFit.activeVar = queue.pop(0)

    def load_maybe(self):
        """
        Load variable from file or set to initial value of the variable.
        """
        value = read_value_json(self.scale_file, self._name)
        if value is None:
            logging.debug(
                f"Initialize variable {self._name}' to {self.variable.numpy():.3f}"
            )
        else:
            self._fitted = True
            logging.debug(f"Set scale factor {self._name} : {value}")
            with torch.no_grad():
                self.variable.copy_(torch.tensor(value))


class AutoScaleFit(AutomaticFit):
    """
    Class to automatically fit the scaling factors depending on the observed variances.

    Parameters
    ----------
        variable: torch.Tensor
            Variable to fit.
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, variable, scale_file, name):
        super().__init__(variable, scale_file, name)

        if not self._fitted:
            self._init_stats()

    def _init_stats(self):
        self.variance_in = 0
        self.variance_out = 0
        self.nSamples = 0

    @torch.no_grad()
    def observe(self, x, y):
        """
        Observe variances for input x and output y.
        The scaling factor alpha is calculated s.t. Var(alpha * y) ~ Var(x)
        """
        if self._fitted:
            return

        # only track stats for current variable
        if AutomaticFit.activeVar == self:
            nSamples = y.shape[0]
            self.variance_in += (
                torch.mean(torch.var(x, dim=0)).to(dtype=torch.float32)
                * nSamples
            )
            self.variance_out += (
                torch.mean(torch.var(y, dim=0)).to(dtype=torch.float32)
                * nSamples
            )
            self.nSamples += nSamples

    @torch.no_grad()
    def fit(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if AutomaticFit.activeVar == self:
            if self.variance_in == 0:
                raise ValueError(
                    f"Did not track the variable {self._name}. Add observe calls to track the variance before and after."
                )

            # calculate variance preserving scaling factor
            self.variance_in = self.variance_in / self.nSamples
            self.variance_out = self.variance_out / self.nSamples

            ratio = self.variance_out / self.variance_in
            value = torch.sqrt(1 / ratio)
            logging.info(
                f"Variable: {self._name}, "
                f"Var_in: {self.variance_in.item():.3f}, "
                f"Var_out: {self.variance_out.item():.3f}, "
                f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
            )

            # set variable to calculated value
            self.variable.copy_(self.variable * value)
            update_json(
                self.scale_file, {self._name: float(self.variable.item())}
            )
            self.set_next_active()  # set next variable in queue to active


class ScalingFactor(torch.nn.Module):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference input x_ref is preserved.

    Parameters
    ----------
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
        name: str
            Name of the scaling factor
    """

    def __init__(self, scale_file, name, device=None):
        super().__init__()

        self.scale_factor = torch.nn.Parameter(
            torch.tensor(1.0, device=device), requires_grad=False
        )
        self.autofit = AutoScaleFit(self.scale_factor, scale_file, name)

    def forward(self, x_ref, y):
        y = y * self.scale_factor
        self.autofit.observe(x_ref, y)

        return y
