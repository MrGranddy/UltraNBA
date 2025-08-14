import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm
from scipy.stats import wasserstein_distance


def rotation_translation_errors(original_poses, perturbed_poses):
    """
    Vectorized calculation of rotation and translation errors.

    Parameters:
        original_poses (np.ndarray): (N, 3, 4) array of original poses.
        perturbed_poses (np.ndarray): (N, 3, 4) array of perturbed poses.

    Returns:
        dict: Statistics of translation and rotation errors.
    """
    assert original_poses.shape == perturbed_poses.shape

    org_rot = original_poses[:, :3, :3]
    pert_rot = perturbed_poses[:, :3, :3]
    org_trans = original_poses[:, :3, 3]
    pert_trans = perturbed_poses[:, :3, 3]

    # Vectorized computation of relative rotation matrices
    R_rel = np.einsum("nij,njk->nik", pert_rot, np.transpose(org_rot, (0, 2, 1)))

    # Efficiently compute rotation angles from rotation matrices using trace
    cos_theta = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
    rot_errors = np.degrees(np.arccos(cos_theta))

    # Compute translation errors efficiently
    trans_errors = np.linalg.norm(pert_trans - org_trans, axis=1)

    return {
        "translation_mean": np.mean(trans_errors),
        "translation_std": np.std(trans_errors),
        "rotation_mean": np.mean(rot_errors),
        "rotation_std": np.std(rot_errors),
    }


def kabsch_alignment(A, B):
    """
    Computes optimal rotation and translation (rigid alignment) using Kabsch algorithm.

    Parameters:
        A (np.ndarray): (N, 3) Original points.
        B (np.ndarray): (N, 3) Perturbed points to align.

    Returns:
        R (np.ndarray): Optimal rotation matrix (3x3).
        t (np.ndarray): Optimal translation vector (3,).
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    H = (B - centroid_B).T @ (A - centroid_A)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_A - R @ centroid_B

    return R, t


def rotation_translation_errors_procrustes(original_poses, perturbed_poses):
    """
    Computes rotation and translation errors after global alignment with Procrustes, as used in BARF.

    Parameters:
        original_poses (np.ndarray): (N, 3, 4) original poses.
        perturbed_poses (np.ndarray): (N, 3, 4) perturbed/refined poses.

    Returns:
        dict: Statistics of aligned translation and rotation errors.
    """
    assert original_poses.shape == perturbed_poses.shape

    org_rot = original_poses[:, :3, :3]
    pert_rot = perturbed_poses[:, :3, :3]
    org_trans = original_poses[:, :3, 3]
    pert_trans = perturbed_poses[:, :3, 3]

    # Procrustes Alignment for translation
    org_trans_centered = org_trans - np.mean(org_trans, axis=0)
    pert_trans_centered = pert_trans - np.mean(pert_trans, axis=0)

    H = org_trans_centered.T @ pert_trans_centered
    U, _, Vt = np.linalg.svd(H)
    R_global = Vt.T @ U.T

    # Ensure a proper rotation matrix (determinant should be 1)
    if np.linalg.det(R_global) < 0:
        Vt[-1, :] *= -1
        R_global = Vt.T @ U.T

    t_global = np.mean(org_trans, axis=0) - R_global @ np.mean(pert_trans, axis=0)

    # Apply global alignment to perturbed poses
    pert_trans_aligned = (R_global @ pert_trans.T).T + t_global
    pert_rot_aligned = R_global @ pert_rot

    # Compute translation errors (after Procrustes alignment)
    trans_errors = np.linalg.norm(pert_trans_aligned - org_trans, axis=1)

    # Compute relative rotations using proper geodesic rotation metric
    R_rel = np.einsum(
        "nij,njk->nik", pert_rot_aligned, np.transpose(org_rot, (0, 2, 1))
    )

    # Compute geodesic rotation error
    cos_theta = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_errors = np.degrees(np.arccos(cos_theta))

    return {
        "translation_errors": trans_errors,
        "rotation_errors": rot_errors,
    }


def rotation_translation_errors_kabsch(original_poses, perturbed_poses):
    """
    Computes rotation and translation errors after global alignment with Kabsch.

    Parameters:
        original_poses (np.ndarray): (N, 3, 4) original poses.
        perturbed_poses (np.ndarray): (N, 3, 4) perturbed/refined poses.

    Returns:
        dict: Statistics of aligned translation and rotation errors.
    """
    assert original_poses.shape == perturbed_poses.shape

    org_rot = original_poses[:, :3, :3]
    pert_rot = perturbed_poses[:, :3, :3]
    org_trans = original_poses[:, :3, 3]
    pert_trans = perturbed_poses[:, :3, 3]

    # Align translation globally using Kabsch
    R_global, t_global = kabsch_alignment(org_trans, pert_trans)

    # Apply global alignment to perturbed poses
    pert_trans_aligned = (R_global @ pert_trans.T).T + t_global
    pert_rot_aligned = R_global @ pert_rot

    # Compute translation errors after alignment
    trans_errors = np.linalg.norm(pert_trans_aligned - org_trans, axis=1)

    # Compute relative rotations
    R_rel = np.einsum(
        "nij,njk->nik", pert_rot_aligned, np.transpose(org_rot, (0, 2, 1))
    )

    # Efficiently compute rotation angles using trace
    cos_theta = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_errors = np.degrees(np.arccos(cos_theta))

    return {
        "translation_errors": trans_errors,
        "rotation_errors": rot_errors,
    }


def trajectory_wasserstein_distance(original_poses, refined_poses):
    """
    Computes the Earth Mover's Distance (Wasserstein-1) between the original and refined translations.

    Parameters:
        original_poses (np.ndarray): (N, 3, 4) original poses.
        refined_poses (np.ndarray): (N, 3, 4) refined poses.

    Returns:
        float: Wasserstein distance between original and refined translations.
    """
    original_translations = original_poses[:, :3, 3]
    refined_translations = refined_poses[:, :3, 3]

    return np.mean(
        [
            wasserstein_distance(
                original_translations[:, i], refined_translations[:, i]
            )
            for i in range(3)
        ]
    )


def frame_to_frame_variability(poses):
    """
    Computes frame-to-frame transformation variability as a measure of consistency.

    Parameters:
        poses (np.ndarray): (N, 3, 4) poses.

    Returns:
        float: Standard deviation of relative transformations.
    """
    rel_transforms = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.std(rel_transforms)


def rotation_geodesic_distance(original_poses, refined_poses):
    """
    Computes the geodesic distance between original and refined rotations.

    Parameters:
        original_poses (np.ndarray): (N, 3, 4) original poses.
        refined_poses (np.ndarray): (N, 3, 4) refined poses.

    Returns:
        float: Mean geodesic rotation distance.
    """
    original_rotations = original_poses[:, :3, :3]
    refined_rotations = refined_poses[:, :3, :3]

    distances = [
        np.linalg.norm(logm(refined_rotations[i] @ original_rotations[i].T), "fro")
        for i in range(len(original_rotations))
    ]

    return np.mean(distances)


def calculate_total_rot_and_trans_errors(original_poses, perturbed_poses):
    """
    Calculates the rotation and translation errors between the original and perturbed poses.

    Parameters:
        original_poses (np.ndarray): Array of shape (N, 3, 4), original poses.
        perturbed_poses (np.ndarray): Array of shape (N, 3, 4), perturbed poses.

    Returns:
        rot_errors (np.ndarray): Array of shape (N,), rotation errors in degrees
    """

    assert original_poses.shape == perturbed_poses.shape
    N = original_poses.shape[0]

    # Extract rotations and translations
    org_rot = original_poses[:, :3, :3]
    pert_rot = perturbed_poses[:, :3, :3]
    org_trans = original_poses[:, :3, 3]
    pert_trans = perturbed_poses[:, :3, 3]

    # Compute rotation errors using logm (Lie algebra method)
    rot_errors = []
    for i in range(N):
        R_rel = pert_rot[i] @ np.linalg.inv(org_rot[i])  # Relative rotation
        log_R, errest = logm(
            R_rel, disp=False
        )  # Matrix logarithm to get Lie algebra representation
        theta = np.linalg.norm(log_R, "fro") / np.sqrt(
            2
        )  # Frobenius norm maps to rotation angle
        rot_errors.append(np.degrees(theta))  # Convert to degrees

    # Compute translation errors
    trans_errors = np.linalg.norm(pert_trans - org_trans, axis=1)

    return {
        "rotation_error": np.sum(np.abs(rot_errors)),
        "translation_error": np.sum(trans_errors),
    }


def check_valid_transformation(transformation):

    rot_matrix = transformation[:3, :3]

    # Check if the matrix is a valid rotation matrix
    det = np.linalg.det(rot_matrix)
    gram_matrix = rot_matrix.transpose() @ rot_matrix
    identity_matrix = np.eye(3)

    if not np.allclose(gram_matrix, identity_matrix, atol=1e-5):
        print(f"Gram matrix is not close to identity matrix: {gram_matrix}")

    if not np.isclose(det, 1.0, atol=1e-5):
        print(f"Det is not close to 1: {det}")
