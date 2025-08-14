import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


def cumsum_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumsum(..., exclusive=True), as it isn't available in PyTorch.

    Args:
      tensor (torch.Tensor): Tensor whose cumsum (cumulative product, see `torch.cumsum`) along dim=-1
        is to be computed.

    Returns:
      cumsum (torch.Tensor): cumsum of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumsum(..., exclusive=True) (see `tf.math.cumsum` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1) -> Why?
    dim = -1
    # Compute regular cumsum first (this is equivalent to `tf.math.cumsum(..., exclusive=False)`).
    cumsum = torch.cumsum(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumsum = torch.roll(cumsum, 1, dim)
    # Replace the first element by "0" as this is what tf.cumsum(..., exclusive=True) does.
    cumsum[..., 0] = 0.0

    return cumsum


def gaussian_kernel(size: int, mean: float, std: float):
    delta_t = 1
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t

    d1 = torch.distributions.Normal(mean, std)
    d2 = torch.distributions.Normal(mean, std)
    vals_x = d1.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_y = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()

    gauss_kernel = torch.einsum("i,j->ij", vals_x, vals_y)

    return gauss_kernel / torch.sum(gauss_kernel)


def gaussian_kernel_3d(size: int, mean: float, std: float):
    delta_t = 1
    x_cos = np.array(list(range(-size, size + 1)), dtype=np.float32)
    x_cos *= delta_t

    d1 = torch.distributions.Normal(mean, std * 3)
    d2 = torch.distributions.Normal(mean, std)

    vals_x = d1.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_y = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()
    vals_z = d2.log_prob(
        torch.arange(-size, size + 1, dtype=torch.float32) * delta_t
    ).exp()

    # Create 3D Gaussian kernel by taking the outer product
    gauss_kernel_3d = torch.einsum("i,j,k->ijk", vals_x, vals_y, vals_z)

    # Normalize the kernel so that the sum of all elements equals 1
    gauss_kernel_3d = gauss_kernel_3d / torch.sum(gauss_kernel_3d)

    return gauss_kernel_3d


mu_k, sigma_k = 0.0, 1.0
g_kernel = gaussian_kernel(3, mu_k, sigma_k).float().to(device="cuda")
# size = int(3 * sigma_a)  # Cover ±3σ range
# c = 1540  # Speed of sound in soft tissue (m/s)
# f_c = 6e6  # Transducer center frequency (Hz)
# wavelength = c / f_c  # Wavelength in mm
# frequency = 1 / wavelength  # Spatial frequency of modulation
# x_vals = torch.linspace(-size, size, steps=2*size+1)
# g_kernel *= torch.cos(2 * np.pi * frequency * x_vals)


def safe_log(x, eps=1.0e-6):
    return torch.log(torch.clamp(x, min=eps))


def render_method_3(raw, r_max_reflection):
    """
    raw (torch.Tensor): Raw network outputs. [N_rays, N_samples, C].
    z_vals (torch.Tensor): Steps from 0 to 1 in N_samples direction. [N_rays, N_samples].
    """

    N_rays, N_samples, C = raw.shape

    t_vals = torch.linspace(0.0, 1.0, N_samples).to(device="cuda")
    z_vals = t_vals.expand(N_rays, N_samples)

    # Compute 'distance' between each integration time along a ray.
    dists = torch.abs(z_vals[:, :-1] - z_vals[:, 1:])
    dists = torch.cat([dists, dists[:, -1:]], dim=-1)  # [N_rays, N_samples]

    # ATTENUATION
    attenuation_coeff = raw[..., 0]
    attenuation = torch.exp(-attenuation_coeff * dists)
    log_attenuation = safe_log(attenuation)
    log_attenuation_total = cumsum_exclusive(log_attenuation)

    # REFLECTION
    reflection_coeff = (
        torch.sigmoid(raw[..., 1]) * r_max_reflection
    )  # From Magdalena's calculations it should be multiplied with 0.34
    reflection_transmission = 1.0 - reflection_coeff
    log_reflection_transmission = safe_log(reflection_transmission)
    log_reflection_total = cumsum_exclusive(log_reflection_transmission)

    # BACKSCATTERING
    density_coeff = torch.ones_like(reflection_coeff) * 0.75
    scatter_density_distribution = RelaxedBernoulli(
        temperature=0.1, probs=density_coeff
    )
    scatterers_density = scatter_density_distribution.sample()
    amplitude = raw[..., 2]
    scatterers_map = scatterers_density * amplitude

    psf_scatter = (
        F.conv2d(
            scatterers_map[None, None],
            g_kernel[None, None, ...],
            stride=1,
            padding="same",
        )
        .squeeze(0)
        .squeeze(0)
    )

    # Compute remaining intensity at a point n
    log_confidence_maps = log_attenuation_total + log_reflection_total

    # Compute backscattering and reflection parts of the final echo
    b = torch.exp(log_confidence_maps + safe_log(psf_scatter))
    r = torch.exp(log_confidence_maps + safe_log(reflection_coeff))

    # Compute the final echo
    amplification_constant = np.pi
    log_const = np.log1p(amplification_constant)  # Precompute constant
    r_amplified = torch.log1p(amplification_constant * r) * log_const

    intensity_map = b + r_amplified

    return {
        "intensity_map": intensity_map,
        "attenuation_coeff": attenuation_coeff,
        "reflection_coeff": reflection_coeff,
        "attenuation_total": torch.exp(log_attenuation_total),
        "reflection_total": torch.exp(log_reflection_total),
        "scatterers_density": scatterers_density,
        "scatterers_density_coeff": density_coeff,
        "scatter_amplitude": amplitude,
        "b": b,
        "r": r,
        "r_amplified": r_amplified,
        "confidence_maps": torch.exp(log_confidence_maps),
    }


def render_rays_us(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    lindisp=False,
    **kwargs,
):
    """Volumetric rendering.

    Args:
    ray_batch: Tensor of shape [batch_size, ...]. We define rays and do not sample.

    Returns:
    Rendered outputs.
    """

    def raw2outputs(raw, r_max_reflection):
        """Transforms model's predictions to semantically meaningful values."""
        # TODO: add args controlling the rendering method
        ret = render_method_3(
            raw, r_max_reflection
        )  # Assuming render_method_3 is defined elsewhere
        return ret

    ###############################
    # Batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract lower, upper bound for ray distance
    near, far = ray_batch[..., 6], ray_batch[..., 7]  # [N_rays,] each

    # Decide where to sample along each ray
    near = near.view(-1, 1)  # [N_rays, 1]
    far = far.view(-1, 1)  # [N_rays, 1]
    t_vals = (
        torch.linspace(0.0, 1.0, N_samples).to(ray_batch.device).view(1, -1)
    )  # [1, N_samples]
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    z_vals = z_vals.expand(N_rays, N_samples).unsqueeze(-1)  # [N_rays, N_samples, 1]
    rays_o = rays_o.view(-1, 1, 3)  # [N_rays, 1        , 3]
    rays_d = rays_d.view(-1, 1, 3)  # [N_rays, 1        , 3]

    # Points in space to evaluate model at
    origin = rays_o
    step = rays_d * z_vals

    pts = step + origin

    # Evaluate model at each point
    raw = network_query_fn(pts, network_fn)  # [N_rays, N_samples , 3]

    ret = raw2outputs(raw, kwargs["r_max_reflection"])

    return ret
