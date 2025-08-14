import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

from config import config_parser
from load_us import load_us_data
from utils.nerf import get_rays_us_linear


def load_data():

    parser = config_parser()
    args = parser.parse_args()

    print("Setting deterministic behaviour")
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    images, poses, i_test = load_us_data(
        args.datadir, confmap=args.confmap, pose_path=args.pose_path
    )

    if not isinstance(i_test, list):
        i_test = [i_test]

    i_val = i_test
    i_train = np.array(
        [
            i
            for i in np.arange(int(images.shape[0]))
            if (i not in i_test and i not in i_val)
        ]
    )

    print("Test {}, train {}".format(len(i_test), len(i_train)))

    # Cast intrinsics to right types
    # The poses are not normalized. We scale down the space.
    # It is possible to normalize poses and remove scaling.
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth

    images = images.transpose(0, 2, 1)

    H, W = images.shape[1], images.shape[2]
    sw = probe_depth / float(W)
    sh = probe_width / float(H)

    return H, W, sh, sw, images, poses, near, far, args


def create_rays(H, W, sh, sw, c2w, near, far):

    rays_o, rays_d = [], []

    for c in c2w:
        o, d = get_rays_us_linear(H, W, sh, sw, c)
        rays_o.append(o)
        rays_d.append(d)

    # Concatenate ray origins and directions into a single tensor
    rays_o = torch.cat(rays_o, dim=0).float()
    rays_d = torch.cat(rays_d, dim=0).float()

    # Create near and far distance tensors
    near_tensor = torch.full(
        (rays_o.shape[0], 1), near, dtype=torch.float32, device=c2w.device
    )
    far_tensor = torch.full(
        (rays_o.shape[0], 1), far, dtype=torch.float32, device=c2w.device
    )

    # Combine ray data: (origin, direction, near, far)
    rays = torch.cat([rays_o, rays_d, near_tensor, far_tensor], dim=-1)

    return rays


def rays_to_points(ray_batch, N_samples):

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

    z_vals = near * (1.0 - t_vals) + far * t_vals

    z_vals = z_vals.expand(N_rays, N_samples).unsqueeze(-1)  # [N_rays, N_samples, 1]
    rays_o = rays_o.view(-1, 1, 3)  # [N_rays, 1        , 3]
    rays_d = rays_d.view(-1, 1, 3)  # [N_rays, 1        , 3]

    # Points in space to evaluate model at
    origin = rays_o
    step = rays_d * z_vals

    pts = step + origin

    return pts


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    H, W, sh, sw, images, poses, near, far, args = load_data()

    img_i = 42

    target = torch.Tensor(images[img_i]).to(device).unsqueeze(0).unsqueeze(0)
    pose = torch.from_numpy(poses[img_i, :3, :4]).to(device).unsqueeze(0)

    rays = create_rays(H, W, sh, sw, pose, near, far)

    pts = rays_to_points(rays, W)

    print("Row", pts[0, :200:20])
    print("Col", pts[:200:20, 0])
