import torch

from vdsds.rendering.camera import Camera
from vdsds.utils.silhouette import silhouette_loss


def _all_ones_target(h=32, w=32):
    """(1, 4, H, W) target with a fully opaque mask channel."""
    tgt = torch.zeros(1, 4, h, w)
    tgt[:, 3] = 1.0
    return tgt


def _two_vertices():
    return torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])


def test_original_behavior_inverse_disabled():
    # All vertices land inside the (all-ones) mask: the original
    # over-coverage-only loss contributes nothing.
    loss = silhouette_loss(
        _all_ones_target(), [_two_vertices()], [Camera()]
    )
    assert loss.item() == 0.0


def test_inverse_term_enabled_by_weight():
    base = silhouette_loss(
        _all_ones_target(), [_two_vertices()], [Camera()]
    )
    inverse = silhouette_loss(
        _all_ones_target(), [_two_vertices()], [Camera()], in_weight=1.0
    )
    # The in-mask term is strictly positive (projected vertices are
    # never exactly on sampled mask pixel centers).
    assert inverse.item() > 0.0
    assert base.item() == 0.0


def test_out_of_mask_term_present():
    # One vertex projects outside the mask block (border stays empty).
    tgt = torch.zeros(1, 4, 32, 32)
    tgt[:, 3, 8:24, 8:24] = 1.0
    verts = torch.tensor([[5.0, 5.0, 0.0]])
    loss = silhouette_loss(tgt, [verts], [Camera()])
    assert loss.item() > 0.0
