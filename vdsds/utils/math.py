import torch
import math

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180


def rotation_matrix(axis: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    Compute a 3x3 rotation matrix from a rotation axis and angle (in degrees).

    Args:
        axis (torch.Tensor): Rotation axis, shape (3,), does not need to be normalized.
        angle_deg (float): Rotation angle in degrees.

    Returns:
        torch.Tensor: Rotation matrix of shape (3, 3).
    """
    # Normalize axis
    axis = axis / torch.norm(axis)

    # Convert angle to radians
    theta = DEG2RAD * angle_deg

    # Components
    x, y, z = axis
    c = math.cos(theta)
    s = math.sin(theta)
    C = 1 - c

    R = torch.tensor(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=torch.float32,
        device=axis.device,
    )

    return R


def angle_between(
    v1: torch.Tensor, v2: torch.Tensor, rad: bool = False, eps: float = 1e-8
) -> torch.Tensor:
    """
    Returns angle between two vectors in radians or degrees(default).
    """
    v1_norm = torch.norm(v1, dim=-1, keepdim=True).clamp_min(eps)
    v2_norm = torch.norm(v2, dim=-1, keepdim=True).clamp_min(eps)
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm
    dot = (v1_unit * v2_unit).sum(dim=-1).clamp(-1.0, 1.0)
    if rad:
        return torch.acos(dot)
    return RAD2DEG * torch.acos(dot)
