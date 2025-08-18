import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.losses.ssim_loss import SSIMLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from config import config_parser
from load_us import load_us_data
from utils.nerf import (
    compute_loss,
    create_barf,
    img2mse,
    render_us,
)
from utils.test import calculate_total_rot_and_trans_errors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():

    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed == 0:
        print("Setting deterministic behaviour")
        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if args.dataset_type == "us":
        # IT CONVERTS THE POSE TRANSLATION FROM MM TO M ALREADY!!!
        data = load_us_data(args.datadir, pose_path=args.pose_path)

        images = data["images"]
        poses = data["poses"]
        i_test = data["i_test"]

        if "org_poses" not in data:
            org_poses = data["poses"].copy()
        else:
            org_poses = data["org_poses"]

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

    else:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return

    # Cast intrinsics to right types
    # The poses are not normalized. We scale down the space.
    # It is possible to normalize poses and remove scaling.
    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth

    images = images.transpose(0, 2, 1)
    org_poses = org_poses[:, :3, :4]

    H, W = images.shape[1], images.shape[2]
    sw = probe_depth / float(W)
    sh = probe_width / float(H)

    basedir = args.basedir
    expname = args.expname

    # Create tensorboard writer
    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(basedir, "summaries", expname))

    # Create log dir and copy the config file
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # Create barf model

    (
        render_kwargs_train,
        render_kwargs_test,
        start,
        optimizer,
        pose_optim,
        pose_sched,
    ) = create_barf(torch.tensor(poses[:, :3, :4]), args, device=device, mode="train")

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    N_iters = args.n_iters
    print("Begin")
    print("TRAIN views are", i_train)
    print("TEST views are", i_test)
    print("VAL views are", i_val)

    # Losses
    ssim_loss = SSIMLoss(
        spatial_dims=2,
        data_range=1.0,
        kernel_type="gaussian",
        win_size=args.ssim_filter_size,
        k1=0.01,
        k2=0.1,
    )

    losses = {
        "l2": img2mse,
        "ssim": ssim_loss,
        "lncc": LocalNormalizedCrossCorrelationLoss(spatial_dims=2),
    }

    perm = np.random.permutation(len(i_train))
    perm_idx = 0

    start = start + 1
    for i in trange(start, N_iters + 1):

        time0 = time.time()

        if args.no_freq_adjustment:
            render_kwargs_train["network_fn"].progress.data.fill_(1.0)
        else:
            render_kwargs_train["network_fn"].progress.data.fill_(i / N_iters)

        img_i = perm[perm_idx]
        perm_idx += 1

        if perm_idx == len(i_train):
            perm = np.random.permutation(len(i_train))
            perm_idx = 0

        target = torch.Tensor(images[img_i]).to(device).unsqueeze(0).unsqueeze(0)
        pose = render_kwargs_train["pose_refine"].get_pose(int(img_i)).unsqueeze(0)

        #####  Core optimization loop  #####
        rendering_output = render_us(
            H, W, sh, sw, c2w=pose, chunk=args.chunk, retraw=True, **render_kwargs_train
        )
        output_image = rendering_output["intensity_map"]

        # Add batch and channel to target because pred target is [H, W] and target is [1, 1, H, W]
        # I am not sure if this is necessary but lots of loss functions expect this so, doesn't hurt
        output_image = output_image.unsqueeze(0).unsqueeze(0)

        pose_optim.zero_grad()

        if args.warmup_pose:
            # simple linear warmup of pose learning rate
            pose_optim.param_groups[0]["lr_orig"] = pose_optim.param_groups[0][
                "lr"
            ]  # cache the original learning rate
            pose_optim.param_groups[0]["lr"] *= min(1, i / args.warmup_pose)

        optimizer.zero_grad()

        loss = compute_loss(output_image, target, args, losses)

        total_loss = 0.0
        for loss_value in loss.values():
            tmp = loss_value[0] * loss_value[1]
            total_loss += tmp

        if type(total_loss) != torch.Tensor:
            raise ValueError("Loss is not a tensor: Problem with loss calculation")

        total_loss.backward()
        optimizer.step()
        pose_optim.step()
        pose_sched.step()

        dt = time.time() - time0

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        if args.tensorboard:
            writer.add_scalar("Loss/total_loss", total_loss.item(), i)
            for k, v in loss.items():
                writer.add_scalar(f"Loss/{k}", v[1].item(), i)
            writer.add_scalar("Learning rate", new_lrate, i)

        dt = time.time() - time0
        if (i + 1) % args.i_print == 0:

            rendering_path = os.path.join(basedir, expname, "train_rendering")
            os.makedirs(
                os.path.join(basedir, expname, "train_rendering"), exist_ok=True
            )

            print(f"Step: {i+1}, Loss: {total_loss.item()}, Time: {dt}", flush=True)  # type: ignore
            detailed_loss_string = ", ".join(
                [f"{k}: {v[1].detach().cpu().item()}" for k, v in loss.items()]
            )
            print(detailed_loss_string, flush=True)

            plt.figure(figsize=(16, 8))
            for j, m in enumerate(rendering_output):

                plt.subplot(3, 4, j + 1)
                plt.title(m)
                plt.imshow(rendering_output[m].detach().cpu().numpy().T)

            plt.subplot(3, 4, 12)
            plt.title("Target")
            plt.imshow(target.detach().cpu().numpy()[0, 0].T)

            plt.savefig(
                os.path.join(rendering_path, "{:08d}.png".format(i + 1)),
                bbox_inches="tight",
                dpi=200,
            )
            plt.close()

            with torch.no_grad():
                # Calculate transform errors
                corrected_poses = np.zeros_like(org_poses)
                for j in range(corrected_poses.shape[0]):
                    corrected_poses[j] = (
                        render_kwargs_train["pose_refine"]
                        .get_pose(j)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                errors = calculate_total_rot_and_trans_errors(
                    org_poses, corrected_poses
                )
                for error, val in errors.items():
                    if args.tensorboard:
                        writer.add_scalar(f"Pose Errors/{error}", val, i)

                    print(error, "{:.2f}".format(val), end=" ", flush=True)
                print(flush=True)

        if (i + 1) % args.i_weights == 0:
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i + 1))
            torch.save(
                {
                    "global_step": i,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "pose_optim_state_dict": pose_optim.state_dict(),
                    "pose_sched_state_dict": pose_sched.state_dict(),
                    "pose_refine_state_dict": render_kwargs_train[
                        "pose_refine"
                    ].state_dict(),
                },
                path,
            )
            # print('Saved checkpoints at', path)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    train()
