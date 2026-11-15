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
        _all_ones_target(), [_two_vertices()], [Camera(H=32, W=32)]
    )
    assert loss.item() == 0.0


def test_inverse_term_enabled_by_weight():
    base = silhouette_loss(
        _all_ones_target(), [_two_vertices()], [Camera(H=32, W=32)]
    )
    inverse = silhouette_loss(
        _all_ones_target(), [_two_vertices()], [Camera(H=32, W=32)], in_weight=1.0
    )
    # The under-coverage term is strictly positive: mask pixels are
    # never exactly on projected vertex centers.
    assert inverse.item() > 0.0
    assert base.item() == 0.0


def test_out_of_mask_term_present():
    # One vertex projects outside the mask block (border stays empty)
    # but inside the frame.
    tgt = torch.zeros(1, 4, 32, 32)
    tgt[:, 3, 8:24, 8:24] = 1.0
    verts = torch.tensor([[-0.35, 0.0, 0.0]])  # projects to ~pixel (6.3, 16)
    loss = silhouette_loss(tgt, [verts], [Camera(H=32, W=32)])
    assert loss.item() > 0.0


def test_inverse_term_penalizes_uncovered_mask():
    """The inverse term is a mask-to-vertex under-coverage chamfer.

    A mask region far from every projected vertex must cost much more
    than one the vertex projection covers. (The original commented-out
    vertex-to-mask form cannot express this: an in-mask vertex is
    always within the mask's sampling spacing of a mask pixel.)
    """
    cam = Camera(H=32, W=32)
    verts = torch.tensor([[0.0, 0.0, 0.0]])
    proj = cam.project_w2i(verts)
    col = int(proj[0, 0].round())
    row = int(proj[0, 1].round())

    covered = torch.zeros(1, 4, 32, 32)
    covered[:, 3, row - 2 : row + 3, col - 2 : col + 3] = 1.0
    uncovered = torch.zeros(1, 4, 32, 32)
    uncovered[:, 3, 0:5, col - 2 : col + 3] = 1.0  # top edge, far from the vertex

    l_covered = silhouette_loss(covered, [verts], [cam], in_weight=1.0)
    l_uncovered = silhouette_loss(uncovered, [verts], [cam], in_weight=1.0)
    assert l_covered.item() < l_uncovered.item() / 10.0
    assert l_uncovered.item() > 0.0


def test_non_square_samples_mask_at_projected_pixel():
    """Regression: the mask must be sampled at the projected (row=y, col=x).

    uv_sample expects (row, col) coordinates while project_w2i yields
    (x, y); passing the projection through unswapped sampled the mask at
    the transposed pixel, which only went unnoticed on square images
    with symmetric masks.

    A row-band mask on a non-square image with a full row of vertices:
    every vertex projects into the band (loss 0), but transposed
    sampling reads rows from the x coordinate spanning the whole image
    (loss > 0).
    """
    cam = Camera(H=32, W=16)
    ts = torch.linspace(-0.5, 0.5, 11)
    verts = torch.stack([ts, torch.zeros(11), torch.zeros(11)], dim=1)
    proj = cam.project_w2i(verts)
    row = int(proj[0, 1].round())

    tgt = torch.zeros(1, 4, 32, 16)
    tgt[:, 3, row - 2 : row + 3, :] = 1.0
    loss = silhouette_loss(tgt, [verts], [cam])
    assert loss.item() == 0.0
