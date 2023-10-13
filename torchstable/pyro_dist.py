import torch

from pyro.distributions import Stable

from .pdf import _zeta
from .pdf import stable_standard_density, EPSILON
from .integrator import Batch1DIntegrator


class StableWithLogProb(Stable):
    def __init__(
        self,
        stability,
        skew,
        scale=1,
        loc=0,
        coords="S0",
        validate_args=None,
        integrator=None,
        integration_N_gridpoints=501,
        use_compiled_integrate=True,
    ):
        super().__init__(stability, skew, scale, loc, coords, validate_args)
        if integrator is None:
            self._integrator = Batch1DIntegrator()
        else:
            self._integrator = integrator
        self._integration_N_gridpoints = integration_N_gridpoints
        self.use_compiled_integrate = use_compiled_integrate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        x = (value - self.loc) / self.scale
        p = stable_standard_density(
            x,
            self.stability,
            self.skew,
            self._integrator,
            self.coords,
            self._integration_N_gridpoints,
            compiled_integrate=self.use_compiled_integrate,
        )
        p = torch.where(p == 0.0, p + EPSILON, p / self.scale)
        return p.log().float()

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(StableWithLogProb, _instance)
        batch_shape = torch.Size(batch_shape)
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand(batch_shape))
        new.coords = self.coords
        new._integrator = self._integrator
        new._integrate_fn = self._integrate_fn
        super(Stable, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
