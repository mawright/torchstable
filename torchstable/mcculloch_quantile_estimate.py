import torch
from torch import Tensor
import torch.nn.functional as F


def interp_2d(
    x_range: Tensor, y_range: Tensor, input: Tensor, x_query: Tensor, y_query: Tensor
):
    assert input.shape == (x_range.numel(), y_range.numel())
    assert x_query.numel() == y_query.numel()

    x_index = torch.argmax((x_range >= x_query.unsqueeze(-1)).long(), dim=1)
    y_index = torch.argmax((y_range >= y_query.unsqueeze(-1)).long(), dim=1)

    x = torch.stack(
        [x_range[x_index] - x_query, x_query - x_range[x_index - 1]], dim=-1
    )
    y = torch.stack(
        [y_range[y_index] - y_query, y_query - y_range[y_index - 1]], dim=-1
    )
    f = torch.stack(
        [
            input[x_index - 1, y_index - 1],
            input[x_index - 1, y_index],
            input[x_index, y_index - 1],
            input[x_index, y_index],
        ],
        -1,
    ).reshape(-1, 2, 2)

    unscaled = torch.einsum("bi,bij,bj->b", x, f, y)
    return unscaled / (
        (x_range[x_index] - x_range[x_index - 1])
        * (y_range[y_index] - y_range[y_index - 1])
    )


def quantile_estimate(data):
    dtype = data.dtype
    device = data.device

    nu_alpha_range = torch.tensor(
        [2.439, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.5, 4, 5, 6, 8, 10, 15, 25],
        device=device,
        dtype=dtype,
    )
    nu_beta_range = torch.tensor(
        [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1], device=device, dtype=dtype
    )

    alpha_table = torch.tensor(
        [
            [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
            [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
            [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
            [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
            [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
            [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
            [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
            [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
            [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
            [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
            [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
            [0.896, 0.892, 0.884, 0.883, 0.855, 0.823, 0.769],
            [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
            [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.597],
            [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513],
        ],
        device=device,
        dtype=dtype,
    ).T

    beta_table = torch.tensor(
        [
            [0, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
            [0, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
            [0, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
            [0, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
            [0, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
            [0, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
            [0, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
            [0, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
            [0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
            [0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
            [0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
            [0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
            [0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
            [0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
            [0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274],
        ],
        device=device,
        dtype=dtype,
    ).T

    alpha_range = torch.tensor(
        [2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5][
            ::-1
        ],
        device=device,
        dtype=dtype,
    )
    beta_range = torch.tensor([0, 0.25, 0.5, 0.75, 1], device=device, dtype=dtype)

    nu_c_table = (
        torch.tensor(
            [
                [1.908, 1.908, 1.908, 1.908, 1.908],
                [1.914, 1.915, 1.916, 1.918, 1.921],
                [1.921, 1.922, 1.927, 1.936, 1.947],
                [1.927, 1.930, 1.943, 1.961, 1.987],
                [1.933, 1.940, 1.962, 1.997, 2.043],
                [1.939, 1.952, 1.988, 2.045, 2.116],
                [1.946, 1.967, 2.022, 2.106, 2.211],
                [1.955, 1.984, 2.067, 2.188, 2.333],
                [1.965, 2.007, 2.125, 2.294, 2.491],
                [1.980, 2.040, 2.205, 2.435, 2.696],
                [2.000, 2.085, 2.311, 2.624, 2.973],
                [2.040, 2.149, 2.461, 2.886, 3.356],
                [2.098, 2.244, 2.676, 3.265, 3.912],
                [2.189, 2.392, 3.004, 3.844, 4.775],
                [2.337, 2.634, 3.542, 4.808, 6.247],
                [2.588, 3.073, 4.534, 6.636, 9.144],
            ],
            device=device,
            dtype=dtype,
        )
        .flipud()
        .T
    )

    nu_zeta_table = (
        torch.tensor(
            [
                [0, 0.000, 0.000, 0.000, 0.000],
                [0, -0.017, -0.032, -0.049, -0.064],
                [0, -0.030, -0.061, -0.092, -0.123],
                [0, -0.043, -0.088, -0.132, -0.179],
                [0, -0.056, -0.111, -0.170, -0.232],
                [0, -0.066, -0.134, -0.206, -0.283],
                [0, -0.075, -0.154, -0.241, -0.335],
                [0, -0.084, -0.173, -0.276, -0.390],
                [0, -0.090, -0.192, -0.310, -0.447],
                [0, -0.095, -0.208, -0.346, -0.508],
                [0, -0.098, -0.223, -0.380, -0.576],
                [0, -0.099, -0.237, -0.424, -0.652],
                [0, -0.096, -0.250, -0.469, -0.742],
                [0, -0.089, -0.262, -0.520, -0.853],
                [0, -0.078, -0.272, -0.581, -0.997],
                [0, -0.061, -0.279, -0.659, -1.198],
            ],
            device=device,
            dtype=dtype,
        )
        .flipud()
        .T
    )

    p05, p25, p50, p75, p95 = torch.quantile(
        data,
        torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=device, dtype=dtype),
    )

    nu_alpha = (p95 - p05) / (p75 - p25)
    nu_beta = (p95 + p05 - 2 * p50) / (p95 - p05)

    if nu_alpha.ndim == 0:
        nu_alpha = nu_alpha.unsqueeze(0)
    if nu_beta.ndim == 0:
        nu_beta = nu_beta.unsqueeze(0)

    def psi_1(nu_beta, nu_alpha):
        return torch.where(
            nu_beta > 0,
            interp_2d(nu_beta_range, nu_alpha_range, alpha_table, nu_beta, nu_alpha),
            -interp_2d(nu_beta_range, nu_alpha_range, alpha_table, -nu_beta, nu_alpha),
        )

    def psi_2(nu_beta, nu_alpha):
        return torch.where(
            nu_beta > 0,
            interp_2d(nu_beta_range, nu_alpha_range, beta_table, nu_beta, nu_alpha),
            -interp_2d(nu_beta_range, nu_alpha_range, beta_table, -nu_beta, nu_alpha),
        )

    def phi_3(beta, alpha):
        return torch.where(
            beta > 0,
            interp_2d(beta_range, alpha_range, nu_c_table, beta, alpha),
            interp_2d(beta_range, alpha_range, nu_c_table, -beta, alpha),
        )

    def phi_5(beta, alpha):
        return torch.where(
            beta > 0,
            interp_2d(beta_range, alpha_range, nu_zeta_table, beta, alpha),
            interp_2d(beta_range, alpha_range, nu_zeta_table, -beta, alpha),
        )

    _eps = torch.finfo(dtype).eps

    nu_alpha_geq_2439 = nu_alpha >= 2.439
    alpha = torch.where(
        nu_alpha_geq_2439, torch.clamp(psi_1(nu_beta, nu_alpha), _eps, 2.0), 2.0
    )
    beta = torch.where(
        nu_alpha_geq_2439,
        torch.clamp(psi_2(nu_beta, nu_alpha), -1.0, 1.0),
        torch.sign(nu_beta),
    )
    gamma = (p75 - p25) / phi_3(beta, alpha)
    zeta = p50 + gamma * phi_5(beta, alpha)
    # delta1 = torch.where(
    # alpha != 1.,
    # zeta - beta * gamma * torch.tan(torch.pi * alpha / 2.),
    # zeta
    # )

    delta0 = torch.where(
        alpha != 1.0, zeta, zeta + 2 * beta * gamma * torch.log(gamma) / torch.pi
    )

    # Note: different return order than scipy
    return torch.cat([alpha, beta, gamma, delta0], -1)


def quantile_loss(data, parametric_estimates):
    return F.mse_loss(parametric_estimates, quantile_estimate(data), reduction="none").sum(-1)


def cosine_decay(lr_0, t_max, t):
    out = .5 * lr_0 * (1 + torch.cos(t / t_max * torch.tensor(torch.pi))),
    out[t >= t_max] = 0.0
    return out
