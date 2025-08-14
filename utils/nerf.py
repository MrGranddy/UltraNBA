import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BARF, NeRF, PoseRefine
from rendering import render_rays_us

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log10(x)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def load_experiment_args(filepath: str) -> dict:
    """
    Load experiment arguments from a text file.

    Args:
        filepath (str): Path to the arguments file.

    Returns:
        dict: Parsed arguments as a dictionary.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    return {
        key.strip(): value.strip()
        for key, value in (line.strip().split("=") for line in lines)
    }


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(
                0.0, max_freq, steps=N_freqs, device=self.kwargs["device"]
            )
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0**max_freq, steps=N_freqs, device=self.kwargs["device"]
            )

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, device):

    embed_kwargs = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
        "device": device,
    }

    embedder_obj = Embedder(**embed_kwargs)

    return embedder_obj.embed, embedder_obj.out_dim


def get_rays_us_linear(H, W, sh, sw, c2w):
    t = c2w[:3, -1]
    R = c2w[:3, :3]

    y = torch.arange(-H / 2, H / 2, dtype=torch.float32, device=c2w.device) * sh * -1
    x = torch.ones_like(y) * (-W / 2) * sw
    z = torch.zeros_like(x)

    origin_base = torch.stack([x, y, z], dim=1).to(c2w.device)

    origin_rotated = R @ torch.t(origin_base)
    ray_o_r = torch.t(origin_rotated)

    rays_o = ray_o_r + t

    dirs_base = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=c2w.device)
    dirs_r = torch.t(R @ dirs_base.unsqueeze(-1))

    rays_d = dirs_r.expand_as(rays_o)

    return rays_o, rays_d


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    return ret


def run_network(inputs, fn, embed_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = inputs.view(-1, inputs.shape[-1])

    embedded = embed_fn(inputs_flat)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def run_barf_network(inputs, fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    outputs_flat = batchify(fn, netchunk)(inputs_flat)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays_us(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def create_nerf(args, device, mode="train"):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, device)

    output_ch = args.output_ch
    skips = [4]
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
    ).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, network_fn: run_network(
        inputs, network_fn, embed_fn=embed_fn, netchunk=args.netchunk
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]

        if mode == "train":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # remove paramerts "views_linears.0.weight", "views_linears.0.bias" from state_dict
        # if exists, this is for compatibility with the old code
        new_state_dict = {}
        for k, v in ckpt["network_fn_state_dict"].items():
            if "views_linears.0" not in k:
                new_state_dict[k] = v

        # Load model
        model.load_state_dict(new_state_dict)

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "network_fn": model,
        "r_max_reflection": args.r_max_reflection,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, optimizer


def create_barf(poses: torch.Tensor, args, device, mode="train"):
    """Instantiate BARF's MLP model."""

    output_ch = args.output_ch
    skips = [4]
    input_ch = 3

    model = BARF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        L=args.L,
    ).to(device)

    pose_refine = PoseRefine(poses=poses, mode=mode).to(device)

    network_query_fn = lambda inputs, network_fn: run_barf_network(
        inputs, network_fn, netchunk=args.netchunk
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999)
    )

    pose_optim = torch.optim.Adam(pose_refine.parameters(), args.pose_lr)
    gamma = (args.pose_lr_end / args.pose_lr) ** (1.0 / args.n_iters)
    pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_optim, gamma)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    print("Found ckpts", ckpts)
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        pose_optim.load_state_dict(ckpt["pose_optim_state_dict"])
        pose_sched.load_state_dict(ckpt["pose_sched_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        pose_refine.load_state_dict(ckpt["pose_refine_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "network_fn": model,
        "pose_refine": pose_refine,
        "ckpt": ckpt if len(ckpts) > 0 else None,
        "r_max_reflection": args.r_max_reflection,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}

    return (
        render_kwargs_train,
        render_kwargs_test,
        start,
        optimizer,
        pose_optim,
        pose_sched,
    )


def render_us(
    H,
    W,
    sh,
    sw,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    near=0.0,
    far=55.0 * 0.001,
    **kwargs
):
    """
    Render ultrasound rays.

    Args:
        H (int): Image height.
        W (int): Image width.
        sh (float): Scaling factor for height.
        sw (float): Scaling factor for width.
        chunk (int, optional): Chunk size for batch processing (default: 32K).
        rays (tuple, optional): Precomputed (rays_o, rays_d) tensors.
        c2w (list, optional): List of 4x4 camera-to-world matrices.
        near (float, optional): Near clipping plane (default: 0.0).
        far (float, optional): Far clipping plane (default: 55.0 mm).
        **kwargs: Additional parameters for batchify_rays.

    Returns:
        dict: Rendered output from batchify_rays.
    """

    if rays is None and c2w is None:
        raise ValueError("Either 'rays' or 'c2w' must be provided.")

    rays_o, rays_d = [], []

    if c2w is not None:
        for c in c2w:
            o, d = get_rays_us_linear(H, W, sh, sw, c)
            rays_o.append(o)
            rays_d.append(d)
    elif rays:
        # Use provided ray batch
        rays_o, rays_d = rays
    else:
        raise ValueError("Either 'c2w' or 'rays' must be provided.")

    # Concatenate ray origins and directions into a single tensor
    rays_o = torch.cat(rays_o, dim=0).float()
    rays_d = torch.cat(rays_d, dim=0).float()

    # Create near and far distance tensors
    near_tensor = torch.full((rays_o.shape[0], 1), near, dtype=torch.float32).to(
        rays_o.device
    )
    far_tensor = torch.full((rays_o.shape[0], 1), far, dtype=torch.float32).to(
        rays_o.device
    )

    # Combine ray data: (origin, direction, near, far)
    rays = torch.cat([rays_o, rays_d, near_tensor, far_tensor], dim=-1)

    # Render rays in chunks and return result
    return batchify_rays(rays, chunk=chunk, N_samples=W, **kwargs)


def compute_loss(output, target, args, losses):
    loss = {}

    if args.loss == "l2":
        l2_intensity_loss = losses["l2"](output, target)
        loss["l2"] = (1.0, l2_intensity_loss)
    elif args.loss == "ssim":
        ssim_intensity_loss = losses["ssim"](output, target)
        loss["ssim"] = (args.ssim_lambda, ssim_intensity_loss)
        l2_intensity_loss = img2mse(output, target)
        loss["l2"] = ((1 - args.ssim_lambda), l2_intensity_loss)

    return loss


def compute_regularization(rendering_output, reg_funcs, weights=(0.01, 0.00001, 0.34)):

    lncc = reg_funcs["lncc"]
    lncc_w, tv_w, refl_max = weights

    N_rays, N_samples = rendering_output["scatter_amplitude"].shape[-2:]

    scatter_amplitude = (
        rendering_output["scatter_amplitude"]
        .transpose(-2, -1)
        .view(1, 1, N_samples, N_rays)
    )
    attenuation_coeff = (
        rendering_output["attenuation_coeff"]
        .transpose(-2, -1)
        .view(1, 1, N_samples, N_rays)
    )
    reflection_coeff = (
        rendering_output["reflection_coeff"]
        .transpose(-2, -1)
        .view(1, 1, N_samples, N_rays)
    )

    lcc_penalty_scatter_attenuation = lncc(scatter_amplitude, attenuation_coeff)

    reg = {}

    reg["lcc_penalty"] = (
        lncc_w,
        lcc_penalty_scatter_attenuation + 1,
    )  # add one to make sure it is not negative

    norm_scatter_amplitude = scatter_amplitude - scatter_amplitude.min()
    norm_scatter_amplitude = norm_scatter_amplitude / norm_scatter_amplitude.max()

    dy_ampl = norm_scatter_amplitude[:, :, :, 1:] - norm_scatter_amplitude[:, :, :, :-1]
    dx_ampl = norm_scatter_amplitude[:, :, 1:, :] - norm_scatter_amplitude[:, :, :-1, :]

    dy_ampl = F.pad(dy_ampl, (0, 1), "constant", 0)
    dx_ampl = F.pad(dx_ampl, (0, 0, 0, 1), "constant", 0)

    # Calculate TV penalties
    total_variation_penalty_y_ampl = torch.mean(
        (refl_max - reflection_coeff) * torch.abs(dy_ampl)
    )
    total_variation_penalty_x_ampl = torch.mean(
        (refl_max - reflection_coeff) * torch.abs(dx_ampl)
    )

    amplitude_tv_penalty = (
        total_variation_penalty_x_ampl + total_variation_penalty_y_ampl
    )
    reg["tv_penalty"] = (tv_w, amplitude_tv_penalty)

    return reg
