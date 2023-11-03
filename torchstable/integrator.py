import warnings

import torch
from loguru import logger
from autoray import numpy as anp
from autoray import infer_backend

from torchquad import Gaussian


class Batch1DIntegrator(Gaussian):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_integration_domain_check = True
        self._compiled_integrate_fn = None
        self._compiled_integrate_dim = None
        self._compiled_integrate_N = None

    def _resize_roots(self, integration_domain: torch.Tensor, roots: torch.Tensor):
        assert roots.ndim == 1
        assert integration_domain.ndim == 2
        assert integration_domain.shape[-1] == 2

        roots = roots.to(integration_domain.device)

        a = integration_domain[:, 0]
        b = integration_domain[:, 1]

        roots = roots.unsqueeze(-1)
        out = ((b - a) / 2) * roots + ((a + b) / 2)
        logger.debug(f"resize_roots output: {out}")
        return out

    def integrate(self, fn, dim, N, integration_domain=None, backend="torch"):
        assert dim == 1
        assert backend == "torch"

        if integration_domain.ndim == 1:
            integration_domain = integration_domain.reshape(1, 2)

        grid_points, hs, n_per_dim = self.calculate_grid(N, integration_domain)

        logger.debug("Evaluating integrand on the grid.")
        weights = self._weights(n_per_dim, dim, backend).to(grid_points.device)
        function_values, num_points = self.evaluate_integrand(
            fn, grid_points, weights=weights
        )
        self._nr_of_fevals = num_points

        return self.calculate_result(
            function_values, dim, n_per_dim, hs, integration_domain
        )

    def calculate_grid(self, N, integration_domain):
        N = self._adjust_N(dim=1, N=N)

        grid = self._grid_func(
            integration_domain,
            N,
            requires_grad=integration_domain.requires_grad,
            backend="torch",
        )

        h = grid[1] - grid[0]

        return grid, h, N

    @staticmethod
    def _apply_composite_rule(cur_dim_areas, dim, hs, domain):
        return 0.5 * (domain[:, 1] - domain[:, 0]) * cur_dim_areas.sum(-1)

    @staticmethod
    def evaluate_integrand(fn, points, weights=None, args=None):
        """Evaluate the integrand function at the passed points

        Args:
            fn (function): Integrand function
            points (backend tensor): Integration points
            weights (backend tensor, optional): Integration weights. Defaults to None.
            args (list or tuple, optional): Any arguments required by the function. Defaults to None.

        Returns:
            backend tensor: Integrand function output
            int: Number of evaluated points

        Note: This method is copied from torchquad BaseIntegrator to add a torch device transfer
        """
        num_points = points.shape[0]

        if args is None:
            args = ()

        result = fn(points, *args)

        if infer_backend(result) != infer_backend(points):
            warnings.warn(
                "The passed function's return value has a different numerical backend than the passed points. Will try to convert. Note that this may be slow as it results in memory transfers between CPU and GPU, if torchquad uses the GPU."
            )
            result = anp.array(result, like=points)

        num_results = result.shape[0]
        if num_results != num_points:
            raise ValueError(
                f"The passed function was given {num_points} points but only returned {num_results} value(s)."
                f"Please ensure that your function is vectorized, i.e. can be called with multiple evaluation points at once. It should return a tensor "
                f"where first dimension matches length of passed elements. "
            )

        if weights is not None:
            if (
                len(result.shape) > 1
            ):  # if the the integrand is multi-dimensional, we need to reshape/repeat weights so they can be broadcast in the *=
                integrand_shape = anp.array(
                    result.shape[1:],
                    like=infer_backend(points),
                    device=result.device,  # Note: device arg added
                )
                weights = anp.repeat(
                    anp.expand_dims(weights, axis=1), anp.prod(integrand_shape)
                ).reshape((weights.shape[0], *(integrand_shape)))
            result *= weights

        return result, num_points

    def compiled_integrate(self, fn, dim, N, integration_domain, backend="torch"):
        if (
            self._compiled_integrate_fn is None
            or self._compiled_integrate_dim != dim
            or self._compiled_integrate_N != N
        ):
            self._compiled_integrate_fn = self.get_jit_compiled_integrate(
                dim, N, integration_domain, backend
            )
            self._compiled_integrate_dim = dim
            self._compiled_integrate_N = N
        return self._compiled_integrate_fn(fn, integration_domain)

    def get_jit_compiled_integrate(
        self, dim, N=None, integration_domain=None, backend=None
    ):
        """Create an integrate function where the performance-relevant steps except the integrand evaluation are JIT compiled.
        Use this method only if the integrand cannot be compiled.
        The compilation happens when the function is executed the first time.
        With PyTorch, return values of different integrands passed to the compiled function must all have the same format, e.g. precision.

        Args:
            dim (int): Dimensionality of the integration domain.
            N (int, optional): Total number of sample points to use for the integration. See the integrate method documentation for more details.
            integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It can also determine the numerical backend.
            backend (string, optional): Numerical backend. Defaults to integration_domain's backend if it is a tensor and otherwise to the backend from the latest call to set_up_backend or "torch" for backwards compatibility.

        Returns:
            function(fn, integration_domain): JIT compiled integrate function where all parameters except the integrand and domain are fixed
        """
        # If N is None, use the minimal required number of points per dimension
        if N is None:
            N = self._get_minimal_N(dim)

        if backend in ["tensorflow", "jax"]:
            raise ValueError

        elif backend == "torch":
            # Torch requires explicit tracing with example inputs.
            def do_compile(example_integrand):
                import torch

                # Define traceable first and third steps
                def step1(integration_domain):
                    grid_points, hs, n_per_dim = self.calculate_grid(
                        N, integration_domain
                    )
                    return (
                        grid_points,
                        hs,
                        torch.Tensor([n_per_dim]),
                    )  # n_per_dim is constant

                def step3(function_values, hs, integration_domain):
                    return self.calculate_result(
                        function_values, dim, n_per_dim, hs, integration_domain
                    )

                # Trace the first step
                step1 = torch.jit.trace(step1, (integration_domain,))

                # Get example input for the third step
                grid_points, hs, n_per_dim = step1(integration_domain)
                n_per_dim = int(n_per_dim)
                function_values, _ = self.evaluate_integrand(
                    example_integrand,
                    grid_points,
                    weights=self._weights(n_per_dim, dim, backend),
                )

                # Trace the third step
                # Avoid the warnings about a .grad attribute access of a
                # non-leaf Tensor
                if hs.requires_grad:
                    hs = hs.detach()
                    hs.requires_grad = True
                if function_values.requires_grad:
                    function_values = function_values.detach()
                    function_values.requires_grad = True
                step3 = torch.jit.trace(
                    step3, (function_values, hs, integration_domain)
                )

                # Define a compiled integrate function
                def compiled_integrate(fn, integration_domain):
                    grid_points, hs, _ = step1(integration_domain)
                    function_values, _ = self.evaluate_integrand(
                        fn, grid_points, weights=self._weights(n_per_dim, dim, backend)
                    )
                    result = step3(function_values, hs, integration_domain)
                    return result

                return compiled_integrate

            # Do the compilation when the returned function is executed the
            # first time
            compiled_func = [None]

            def lazy_compiled_integrate(fn, integration_domain):
                if compiled_func[0] is None:
                    compiled_func[0] = do_compile(fn)
                return compiled_func[0](fn, integration_domain)

            return lazy_compiled_integrate

        raise ValueError(f"Compilation not implemented for backend {backend }")
