"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from collections import namedtuple
from typing import Optional, Union

import torch


class ScalingFactor(torch.nn.Module):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference input x_ref is preserved.
    """

    def __init__(self):
        super().__init__()

        self.scale_factor = torch.nn.Parameter(
            torch.tensor(1.0), requires_grad=False
        )
        self.fitting_active = False

    def start_fitting(self):
        self.fitting_active = True
        self.variance_in = 0
        self.variance_out = 0
        self.num_samples = 0

    @torch.no_grad()
    def observe(self, x, x_ref=None):
        """
        Observe variances for output x and reference (input) x_ref.
        The scaling factor alpha is chosen s.t. Var(alpha * x) ~ Var(x_ref),
        or, if no x_ref is given, s.t. Var(alpha * x) ~ 1.
        """
        num_samples = x.shape[0]
        self.variance_out += (
            torch.mean(torch.var(x, dim=0)).to(dtype=torch.float32)
            * num_samples
        )
        if x_ref is None:
            self.variance_in += self.variance_out.new_tensor(num_samples)
        else:
            self.variance_in += (
                torch.mean(torch.var(x_ref, dim=0)).to(dtype=torch.float32)
                * num_samples
            )
        self.num_samples += num_samples

    @torch.no_grad()
    def finalize_fitting(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if self.num_samples == 0:
            raise ValueError(
                "A ScalingFactor was not tracked. "
                "Add a forward call to track the variance."
            )

        # calculate variance preserving scaling factor
        self.variance_in = self.variance_in / self.num_samples
        self.variance_out = self.variance_out / self.num_samples

        ratio = self.variance_out / self.variance_in
        value = torch.sqrt(1 / ratio)
        logging.info(
            f"Var_in: {self.variance_in.item():.3f}, "
            f"Var_out: {self.variance_out.item():.3f}, "
            f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
        )

        # set variable to calculated value
        self.scale_factor.copy_(self.scale_factor * value)

        self.fitting_active = False

    def forward(self, x, x_ref=None):
        x = x * self.scale_factor
        if self.fitting_active:
            self.observe(x, x_ref)
        return x


class ScaledModule(torch.nn.Module):
    """
    Automatically register scaling factors for fitting,
    inspired by torch.nn.Module and torch.nn.Parameter.
    """

    def __init__(self):
        super().__init__()
        self._scaling_factors = dict()

    def register_scaling_factor(
        self, name: str, scaling_factor: Optional[ScalingFactor]
    ):
        if "_scaling_factors" not in self.__dict__:
            raise AttributeError(
                "cannot assign scaling_factor before ScaledModule.__init__() call"
            )

        elif not isinstance(name, torch._six.string_classes):
            raise TypeError(
                "scaling_factor name should be a string. "
                "Got {}".format(torch.typename(name))
            )
        elif "." in name:
            raise KeyError('scaling_factor name can\'t contain "."')
        elif name == "":
            raise KeyError('scaling_factor name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._scaling_factors:
            raise KeyError("attribute '{}' already exists".format(name))

        if scaling_factor is None:
            self._scaling_factors[name] = None
        elif not isinstance(scaling_factor, ScalingFactor):
            raise TypeError(
                "cannot assign '{}' object to scaling_factor '{}' "
                "(ScalingFactor or None required)".format(
                    torch.typename(scaling_factor), name
                )
            )
        else:
            self._scaling_factors[name] = scaling_factor

    def scaling_factors(self, prefix: str = "", recurse=True):
        gen = self._named_members(
            lambda module: module._scaling_factors.items()
            if isinstance(module, ScaledModule)
            else {},
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            yield elem

    def __getattr__(self, name: str) -> Union[torch.Tensor, "torch.nn.Module"]:
        if "_scaling_factors" in self.__dict__:
            _scaling_factors = self.__dict__["_scaling_factors"]
            if name in _scaling_factors:
                return _scaling_factors[name]
        return super().__getattr__(name)

    def __setattr__(
        self, name: str, value: Union[torch.Tensor, "torch.nn.Module"]
    ) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        factors = self.__dict__.get("_scaling_factors")
        if isinstance(value, ScalingFactor):
            if factors is None:
                raise AttributeError(
                    "cannot assign scaling factors before ScaledModule.__init__() call"
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_scaling_factor(name, value)
        elif factors is not None and name in factors:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as scaling_factor '{}' "
                    "(ScalingModule or None expected)".format(
                        torch.typename(value), name
                    )
                )
            self.register_scaling_factor(name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._scaling_factors:
            del self._scaling_factors[name]
        else:
            super().__delattr__(name)

    def load_scales(self, scale_factors, strict=True):
        missing_factors = []
        for name, factor in self.scaling_factors():
            if name in scale_factors:
                factor.scale_factor.copy_(scale_factors[name])
            else:
                missing_factors.append(name)

        expected_factors = dict(self.scaling_factors()).keys()
        unexpected_factors = [
            name
            for name in scale_factors.keys()
            if name not in expected_factors
        ]

        if strict:
            error_msg = ""
            if len(unexpected_factors) > 0:
                error_msg += f"Unexpected factors (ignored): {', '.join(unexpected_factors)}.\n"
            if len(missing_factors) > 0:
                error_msg += f"Missing factors (set to 1): {', '.join(missing_factors)}.\n"
            if len(error_msg) > 0:
                logging.warning(
                    "Inconsistencies in loaded scaling factors:\n" + error_msg
                )

        _IncompatibleFactors = namedtuple(
            "IncompatibleFactors", ["unexpected_factors", "missing_factors"]
        )
        return _IncompatibleFactors(unexpected_factors, missing_factors)
