from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from torchquad import BaseIntegrator
from pyro.distributions import Normal

f64info = torch.finfo(torch.float64)
EPSILON = torch.tensor(f64info.eps, dtype=torch.float64)
MAX = torch.tensor(f64info.max, dtype=torch.float64)
MAX_LOG = torch.log(torch.tensor(1e100, dtype=torch.float64))


def stable_standard_density(
    x: Tensor,
    alpha: Tensor,
    beta: Tensor,
    integrator: BaseIntegrator,
    coords: str,
    integration_N_gridpoints=501,
    alpha_near_1_tolerance=0.005,
    x_near_0_tolerance_factor=0.5e-2,
    compiled_integrate=True,
) -> Tensor:
    # Thm 3.3

    if coords == "S0":
        zeta = _zeta(alpha, beta)
        x = x - zeta

    alpha, beta, x = _round_inputs(
        alpha, beta, x, alpha_near_1_tolerance, x_near_0_tolerance_factor
    )

    x = x.double()
    alpha = alpha.double()
    beta = beta.double()

    closed_form_solutions, closed_form_mask = _closed_form_special_cases(alpha, beta, x)

    if x.ndim == 0:
        x = x.unsqueeze(0)
    if alpha.ndim == 0:
        alpha = alpha.unsqueeze(0)
    alpha = alpha.expand_as(x)
    if beta.ndim == 0:
        beta = beta.unsqueeze(0)
    beta = beta.expand_as(x)

    x, beta = _flip_x_beta_if_x_negative(x, beta)

    precomputed_terms = _precomputed_terms(x, alpha, beta)

    lower_bound = _integration_lower_bound(x, alpha, precomputed_terms)
    integration_domain = torch.stack(
        [lower_bound, lower_bound.new_full(lower_bound.shape, torch.pi / 2)], dim=-1
    )

    integrand = partial(
        _integrand,
        x=x,
        alpha=alpha,
        beta=beta,
        precomputed_terms=precomputed_terms,
    )
    if compiled_integrate:
        integral = integrator.compiled_integrate(
            integrand, 1, integration_N_gridpoints, integration_domain
        )
    else:
        integral = integrator.integrate(
            integrand, 1, integration_N_gridpoints, integration_domain
        )

    if integral.ndim == 0:
        integral = integral.unsqueeze(0)

    no_integral_mask = torch.logical_or(
        precomputed_terms.no_integral_mask, closed_form_mask
    )

    integral[no_integral_mask] = 1.0
    integral[integral == 0.0] = EPSILON

    c2 = _c2(x, alpha, beta, precomputed_terms)
    c2[c2 == 0.0] = EPSILON

    out = integral * c2
    if closed_form_mask.any():
        out = torch.where(closed_form_mask, closed_form_solutions, out)
    out = out.clamp_min(1e-100)

    return out


def _round_inputs(alpha, beta, x, alpha_near_1_tolerance, x_near_0_tolerance_factor):
    alpha = torch.where(torch.abs(alpha - 1.0) < alpha_near_1_tolerance, 1.0, alpha)
    # alpha = alpha

    x = torch.where(
        torch.abs(x) < x_near_0_tolerance_factor * alpha ** (1 / alpha), 0.0, x
    )

    return alpha, beta, x


def _closed_form_special_cases(alpha: Tensor, beta: Tensor, x: Tensor):
    out = torch.ones_like(x)

    normal = alpha == 2.0
    levy = torch.logical_and(alpha == 0.5, beta == 1.0)
    cauchy = torch.logical_and(alpha == 1.0, beta == 0.0)
    closed_form_solution_mask = normal.logical_or(levy).logical_or(cauchy)

    if normal.any():
        out = torch.where(
            normal, Normal(0.0, torch.sqrt(torch.as_tensor(2))).log_prob(x).exp(), out
        )

    def _levy_pdf(x):  # from scipy implementation
        return torch.where(
            x < 0,
            0.0,
            1 / torch.sqrt(2 * torch.pi * x) / x * torch.exp(-1 / (2 * x)),
        )

    if levy.any():
        out = torch.where(levy, _levy_pdf(x), out)
    if cauchy.any():
        out = torch.where(cauchy, 1 / (1 + x**2) / torch.pi, out)

    return out, closed_form_solution_mask


def _flip_x_beta_if_x_negative(x: Tensor, beta: Tensor):
    x_negative = x < 0.0

    xnew = torch.where(x_negative, -x, x)
    betanew = torch.where(x_negative, -beta, beta)

    return xnew, betanew


def gamma(tensor: Tensor) -> Tensor:
    return torch.lgamma(tensor).exp()


def _theta_0(alpha: Tensor, beta: Tensor) -> Tensor:
    # Eq (3.22)
    return torch.where(
        alpha == 1.0,
        torch.pi / 2,
        torch.atan(beta * torch.tan(torch.pi * alpha / 2)) / alpha,
    )


@dataclass
class PrecomputedTerms:
    theta_0: Tensor
    alpha_theta0: Tensor
    pi_over_2beta: Tensor
    twobeta_over_pi: Tensor
    xpi_over_2beta: Tensor
    cos_alphatheta0: Tensor
    alpha_over_alphaminus1: Tensor
    no_integral_mask: Tensor
    alpha_neq1_log_term1: Tensor


def _precomputed_terms(x, alpha, beta):
    assert not torch.any(x < 0.0)  # we should have already flipped all the negative x's

    alpha_eq_1 = alpha == 1.0

    theta_0 = _theta_0(alpha, beta)
    alpha_theta0 = theta_0 * alpha
    pi_over_2beta = torch.pi / (2 * beta)
    twobeta_over_pi = 2 * beta / torch.pi
    xpi_over_2beta = x / twobeta_over_pi
    cos_alphatheta0 = torch.cos(alpha * theta_0)
    alpha_over_alphaminus1 = alpha / (alpha - 1 + EPSILON * alpha_eq_1)

    no_integral_mask = torch.logical_or(
        torch.logical_and(alpha != 1.0, x == 0.0),
        torch.logical_and(alpha == 1.0, beta == 0.0),
    )

    alpha_neq1_term1 = cos_alphatheta0.log() * (1 / (alpha - 1 + EPSILON * alpha_eq_1))

    return PrecomputedTerms(
        theta_0=theta_0,
        alpha_theta0=alpha_theta0,
        pi_over_2beta=pi_over_2beta,
        twobeta_over_pi=twobeta_over_pi,
        xpi_over_2beta=xpi_over_2beta,
        cos_alphatheta0=cos_alphatheta0,
        alpha_over_alphaminus1=alpha_over_alphaminus1,
        no_integral_mask=no_integral_mask,
        alpha_neq1_log_term1=alpha_neq1_term1,
    )


def _g(
    theta: Tensor, x: Tensor, alpha: Tensor, beta: Tensor, precomputed_terms: PrecomputedTerms
) -> Tensor:
    cos_theta = theta.cos()

    def alpha_neq_1(
        theta: Tensor,
        x: Tensor,
        alpha: Tensor,
        precomputed_terms: PrecomputedTerms,
        alpha_eq_1_mask: Tensor,
    ):
        mask = alpha_eq_1_mask.logical_or(precomputed_terms.no_integral_mask)

        term1_log = precomputed_terms.alpha_neq1_log_term1

        term2_base = (
            (x + mask)
            * cos_theta
            / torch.sin(precomputed_terms.alpha_theta0 + alpha * theta)
        )
        # term2 = term2_base ** precomputed_terms.alpha_over_alphaminus1
        # assert torch.all(term2 >= 0.0)
        # term2 = term2.clamp_min(EPSILON)
        term2_log = term2_base.log() * precomputed_terms.alpha_over_alphaminus1

        term3 = (
            torch.cos(precomputed_terms.alpha_theta0 + (alpha - 1) * theta) / cos_theta
        )
        assert torch.all(term3 >= 0.0)

        out_log = term1_log + term2_log + term3.log()
        out_log = out_log.clamp_max(MAX_LOG)
        out = ~mask * torch.exp(~mask * out_log)
        if out.isinf().any():
            UserWarning("inf detected in _g for alpha != 1")
        return out

    def alpha_eq_1(
        theta: Tensor, beta: Tensor, precomputed_terms: PrecomputedTerms, alpha_eq_1_mask: Tensor
    ):
        term1 = (1 + theta * precomputed_terms.twobeta_over_pi) / cos_theta
        term2_exponent = (precomputed_terms.pi_over_2beta + theta) * torch.tan(
            theta
        ) - precomputed_terms.xpi_over_2beta
        term2_exponent = term2_exponent.clamp_max(MAX_LOG)
        term2 = torch.exp(alpha_eq_1_mask * term2_exponent)
        return term1 * (term2 * alpha_eq_1_mask)

    # out = torch.where(alpha == 1.0, alpha_eq_1(theta, precomputed_terms), alpha_neq_1(theta, alpha, precomputed_terms))

    alpha_eq_1_mask = alpha == 1.0
    _neq_1 = alpha_neq_1(theta, x, alpha, precomputed_terms, alpha_eq_1_mask)
    _eq_1 = alpha_eq_1(theta, beta, precomputed_terms, alpha_eq_1_mask)

    out = torch.where(alpha_eq_1_mask, _eq_1, _neq_1)
    return out


def _integrand(
    theta: Tensor,
    x: Tensor,
    alpha: Tensor,
    beta: Tensor,
    precomputed_terms: PrecomputedTerms,
):
    assert not torch.any(x < 0.0)  # we should have already flipped all the negative x's

    g = _g(theta, x, alpha, beta, precomputed_terms)
    g = g.clamp(EPSILON, 1e100)
    exp_minus_g = torch.exp(-g)
    out = g * exp_minus_g
    # if out.isnan().any():
    # raise ValueError
    return out


def _c2(
    x: Tensor, alpha: Tensor, beta: Tensor, precomputed_terms: PrecomputedTerms
) -> Tensor:
    assert not torch.any(x < 0.0)  # we should have already flipped all the negative x's

    def alpha_not_1(
        x: Tensor,
        alpha: Tensor,
        precomputed_terms: PrecomputedTerms,
        alpha_eq_1_mask: Tensor,
    ):
        def x_eq_0(alpha, precomputed_terms):
            term1 = gamma(1 + (1 / alpha))
            term2 = precomputed_terms.cos_alphatheta0 ** (1 / alpha)
            term3 = precomputed_terms.theta_0.cos() / torch.pi
            return term1 * term2 * term3

        def x_neq_0(alpha, x, alpha_eq_1_mask):
            return (
                alpha
                / torch.pi
                / torch.abs(
                    alpha - 1 + EPSILON * alpha_eq_1_mask
                )  # avoid divide by 0 by adding epsilon where alpha == 1
                / (x + EPSILON)
            )

        return torch.where(
            x == 0.0,
            x_eq_0(alpha, precomputed_terms),
            x_neq_0(alpha, x, alpha_eq_1_mask),
        )

    def alpha_is_1(x, beta):
        beta_eq_0_mask = beta == 0.0
        return torch.where(
            beta_eq_0_mask,
            1 / (torch.pi * (1 + x**2)),  # repeating the cauchy case...
            1 / (2 * torch.abs(beta) + EPSILON * beta_eq_0_mask),
        )

    alpha_eq_1_mask = alpha == 1.0

    alpha_eq_1 = alpha_is_1(x, beta)
    alpha_neq_1 = alpha_not_1(x, alpha, precomputed_terms, alpha_eq_1_mask)

    return torch.where(alpha_eq_1_mask, alpha_eq_1, alpha_neq_1)


def _integration_lower_bound(x, alpha, precomputed_terms: PrecomputedTerms):
    assert not torch.any(x < 0.0)  # we should have already flipped all the negative x's

    return torch.where(alpha == 1.0, -torch.pi / 2, -precomputed_terms.theta_0)


def _zeta(alpha: Tensor, beta: Tensor) -> Tensor:
    # pg 65
    return torch.where(alpha == 1.0, 0.0, -beta * torch.tan(torch.pi * alpha / 2))
