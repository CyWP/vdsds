import pytest
import torch
from types import SimpleNamespace

from vdsds.utils.view_prompts import (
    ViewPromptSet,
    canonical_directions,
    format_view_weights,
)
from vdsds.utils.deepfloyd import perpneg_combine

T, D = 2, 3  # token count / embed dim of the fake encoder
VIEWS = ["front", "side", "top"]


class FakeEncoder:
    """Deterministic encode_text_2 stand-in (DeepFloyd needs a GPU/model).

    Text i encodes to a constant embedding filled with float(i); the
    trailing uncond row (which the real encode_text_2 appends for
    batch_size=1) is filled with -1. Exact fills make weighted-mix
    assertions exact.
    """

    def encode_text_2(self, texts, negative_prompt=None, batch_size=1):
        rows = [torch.full((T, D), float(i)) for i in range(len(texts))]
        rows.append(torch.full((T, D), -1.0))
        return torch.stack(rows)


def build(mode="mix", num_static=0):
    variant_texts = [f"c{c}v{v}" for c in range(2) for v in range(len(VIEWS))]
    static_texts = [f"s{s}" for s in range(num_static)]
    return ViewPromptSet.build(
        FakeEncoder(), variant_texts, static_texts, VIEWS, mode=mode
    )


def unit(vec):
    return vec / vec.norm()


def test_unknown_view_name_raises():
    with pytest.raises(ValueError):
        canonical_directions(["front", "diagonal"])


def test_deformed_laplacian_smoothing_damps_spikes():
    """Smoothing the displacement field before the Laplacian must reduce
    the penalized energy of a localized spike."""
    import torch

    from vdsds.scripts.train_sds import _cotan_degree_inv

    L = torch.tensor(
        [
            [-2.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, -2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -2.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -2.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, -2.0],
        ]
    )
    dinv = _cotan_degree_inv(type("M", (), {"L_cotan": L.to_sparse().coalesce()}))[
        :, None
    ]
    spike = torch.tensor([[10.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    raw = torch.mm(L, spike)
    D = spike
    for _ in range(2):
        D = D + dinv * torch.mm(L, D)
    smoothed = torch.mm(L, D)
    assert smoothed.pow(2).sum() < raw.pow(2).sum()
    # And the field peak itself is damped by the smoothing steps.
    assert D.abs().max() < spike.abs().max()


    def test_ortho_projection_matrix():
        from vdsds.rendering.camera import Camera

        cam = Camera(H=384, W=384, F=60, orthographic=True, ortho_scale=2.0)
        P = cam.projection
        # Constant w row: no perspective divide.
        assert torch.allclose(P[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
        # Vertical extent 2.0: y = +1 (top) maps to -1 (image-top flip).
        assert torch.isclose(P[1, 1], torch.tensor(-1.0))
        # Horizontal extent 2.0 * (W/H) = 2.0 for a square frame.
        assert torch.isclose(P[0, 0], torch.tensor(1.0))


def test_deform_view_mask():
    from vdsds.deformations.vd_jacobian_full_deform import (
        VDFullJacobianDeformation,
    )

    inst = object.__new__(VDFullJacobianDeformation)
    # None -> full orbit: everything deforms.
    inst.deform_range = None
    deltas = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert torch.allclose(inst.view_mask(deltas), torch.ones(2, 1))

    # Single azimuth 0 (front camera): delta = +Z deforms fully, the
    # +X camera (delta = -X) is gated out.
    inst.deform_range = 0.0
    inst.deform_range_sigma = torch.pi / 8
    front = torch.tensor([[0.0, 0.0, 1.0]])
    side = torch.tensor([[-1.0, 0.0, 0.0]])  # camera at +X -> delta -X
    assert torch.allclose(inst.view_mask(front), torch.ones(1, 1), atol=1e-5)
    assert inst.view_mask(side).item() < 0.01

    # Range [-pi/4, pi/4]: flat 1 inside, taper outside.
    inst.deform_range = (-torch.pi / 4, torch.pi / 4)
    in_range = torch.tensor([[0.0, 0.0, 1.0]])  # theta 0
    at_edge = torch.tensor([[2**0.5 / 2, 0.0, 2**0.5 / 2]])  # theta -pi/4
    outside = torch.tensor([[0.0, 0.0, -1.0]])  # camera at +Z -> theta pi (6 sigma out)
    assert torch.allclose(inst.view_mask(in_range), torch.ones(1, 1))
    assert torch.allclose(inst.view_mask(at_edge), torch.ones(1, 1), atol=1e-5)
    assert inst.view_mask(outside).item() < 0.01

    with pytest.raises(ValueError):
        inst.deform_range = (1.0, 0.0)
        inst.view_mask(front)


def test_view_qualifier_mapping():
    from vdsds.utils.view_prompts import view_qualifier

    assert view_qualifier("front") == "front view"
    assert view_qualifier("frontDiag") == "three-quarter view"
    assert view_qualifier("backDiag") == "rear three-quarter view"


def test_diag_directions_between_canonicals():
    dirs = canonical_directions(["front", "frontDiag", "side", "backDiag", "back"])
    assert torch.allclose(dirs[1], (dirs[0] + dirs[2]) / (dirs[0] + dirs[2]).norm(), atol=1e-6)
    assert torch.allclose(dirs[3], (dirs[2] + dirs[4]) / (dirs[2] + dirs[4]).norm(), atol=1e-6)
    # Even 45-degree spacing over the front half of the orbit.
    dots = (dirs[:-1] * dirs[1:]).sum(-1)
    assert torch.allclose(dots, dots[0])


def test_format_view_weights():
    vs = build()
    weights = vs.view_weights(canonical_directions(["front"]))
    text = format_view_weights(weights[0], vs.views, vs.mode)
    assert text.startswith("view prompts (mix):")
    assert "front 0.79" in text
    assert "side 0.11" in text
    assert "top 0.11" in text


def test_viewer_overlay_hook():
    from vdsds.scripts.train_sds import TrainModelSDS

    inst = object.__new__(TrainModelSDS)
    overlay = inst._viewer_overlay()
    # No viewset yet (or view-dependent prompts off) -> nothing drawn.
    assert overlay(None) is None
    # With a viewset, the overlay formats the weights for the camera.
    vs = build()
    inst.viewset = vs
    inst.model = SimpleNamespace(model=SimpleNamespace(centroid=torch.zeros(3)))
    front_cam = SimpleNamespace(location=torch.tensor([0.0, 0.0, -2.0]))
    text = overlay(front_cam)
    assert text.startswith("view prompts (mix):")
    assert "front 0.79" in text


def test_build_shapes_and_layout():
    vs = build(num_static=2)
    assert vs.embeds.shape == (2, len(VIEWS), T, D)
    assert vs.static_embeds.shape == (2, T, D)
    # (concept, view) fill values are concept-major: c*V + v.
    assert vs.embeds[0, 0, 0, 0].item() == 0.0
    assert vs.embeds[0, 1, 0, 0].item() == 1.0
    assert vs.embeds[1, 2, 0, 0].item() == 5.0
    assert (vs.static_embeds[0] == 6.0).all()
    assert (vs.static_embeds[1] == 7.0).all()
    assert (vs.uncond == -1.0).all()
    assert vs.texts[0] == "c0v0" and vs.static_texts == ["s0", "s1"]


def test_mix_weights_partition_of_unity():
    vs = build()
    dirs = torch.stack(
        [unit(torch.tensor([1.0, 2.0, -3.0])), unit(torch.tensor([0.0, -1.0, 0.0]))]
    )
    weights = vs.view_weights(dirs)
    assert weights.shape == (2, len(VIEWS))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (weights >= 0).all()


def test_canonical_view_dominates():
    vs = build()
    weights = vs.view_weights(canonical_directions(["front"]))
    assert weights.argmax(dim=-1).item() == 0  # front index
    assert weights[0, 0] > 0.7  # near-one-hot at the exact canonical view
    assert weights[0, 0] > 5 * weights[0, 1:].max()


def test_45deg_front_side_equal_mix():
    vs = build()
    # 45 degrees between the front (-Z) and side (+X) directions.
    dirs = unit(torch.tensor([1.0, 0.0, -1.0]))[None]
    weights = vs.view_weights(dirs)
    assert torch.allclose(weights[0, 0], weights[0, 1], rtol=1e-6)
    assert weights[0, :2].min() > weights[0, 2]  # top stays a minor partner


def test_select_mode_onehot():
    vs = build(mode="select")
    exact = vs.view_weights(canonical_directions(["front"]))
    assert torch.equal(exact, torch.tensor([[1.0, 0.0, 0.0]]))
    # 45-degree tie resolves to the first listed view (front).
    tie = vs.view_weights(unit(torch.tensor([1.0, 0.0, -1.0]))[None])
    assert torch.equal(tie, torch.tensor([[1.0, 0.0, 0.0]]))
    # Clearly on the side flank -> one-hot side.
    side = vs.view_weights(unit(torch.tensor([2.0, 0.0, -1.0]))[None])
    assert torch.equal(side, torch.tensor([[0.0, 1.0, 0.0]]))


def test_composite_onehot_is_alpha_split_of_cache_rows():
    vs = build(mode="select")
    front = canonical_directions(["front"])[0]
    dirs = front[None]
    composite = vs.composite(dirs, concept_weights=torch.tensor([[0.3, 0.7]]))
    expected = 0.3 * vs.embeds[0, 0] + 0.7 * vs.embeds[1, 0]
    assert torch.allclose(composite[0], expected)
    # No concept weights -> uniform mix over the concepts.
    uniform = vs.composite(dirs)
    assert torch.allclose(uniform[0], 0.5 * (vs.embeds[0, 0] + vs.embeds[1, 0]))


def test_sds_embeds_layout():
    vs = build(mode="select")
    dirs = canonical_directions(["front"]).repeat(3, 1)  # B=3 identical views
    embeds = vs.sds_embeds(dirs, concept_weights=torch.tensor([[0.3, 0.7]] * 3))
    assert embeds.shape == (6, T, D)
    # First B rows: the per-render composite; last B rows: uncond.
    expected = 0.3 * vs.embeds[0, 0] + 0.7 * vs.embeds[1, 0]
    assert torch.allclose(embeds[:3], expected[None].expand(3, T, D))
    assert torch.allclose(embeds[3:], vs.uncond[None].expand(3, T, D))


def test_bsd_embeds_layout():
    vs = build(num_static=2)  # base prompt branch + BSD blank branch
    B = 2
    dirs = canonical_directions(VIEWS).repeat(B, 1)[:B]  # B=2: front, side
    # 2 concept alphas + 1 base weight column; blank branch carries none.
    attn_alphas = torch.tensor([[0.6, 0.4, 0.1], [0.2, 0.3, 0.1]])
    embeds = vs.bsd_embeds(dirs, attn_alphas, num_funcs=2)
    assert embeds.shape == ((2 + 2 + 1) * B, T, D)
    # Branch-major rows: [c0 x B, c1 x B, static x B, uncond x B]. Each
    # concept branch row (concept c, batch b) must equal alpha[b, c] times
    # the concept's view-weighted cache rows (fake fills c*V + v).
    weights = vs.view_weights(dirs)  # (B, V)
    for c in range(2):
        for b in range(B):
            expected = attn_alphas[b, c] * sum(
                weights[b, v].item() * (c * len(VIEWS) + v)
                for v in range(len(VIEWS))
            )
            assert torch.allclose(
                embeds[(c * B) + b], torch.full((T, D), expected), atol=1e-5
            )
    # Static branches and uncond rows are exact fixed rows.
    for s, text_val in ((0, 6.0), (1, 7.0)):
        assert torch.allclose(embeds[(2 * B) + s * B], torch.full((T, D), text_val))
    assert torch.allclose(embeds[-B:], torch.full((B, T, D), -1.0))


def test_bsd_embeds_alpha_count_mismatch_raises():
    vs = build(num_static=2)
    dirs = canonical_directions(["front"])  # (1, 3)
    with pytest.raises(ValueError):
        vs.bsd_embeds(dirs, torch.ones((1, 2)), num_funcs=2)  # missing base col


def test_save_load_roundtrip(tmp_path):
    vs = build(num_static=1)
    path = tmp_path / "cache" / "view_prompts.pt"
    vs.save(path)
    loaded = ViewPromptSet.load(path)
    assert torch.equal(loaded.embeds, vs.embeds)
    assert torch.equal(loaded.static_embeds, vs.static_embeds)
    assert torch.equal(loaded.uncond, vs.uncond)
    assert loaded.views == vs.views and loaded.mode == vs.mode
    assert loaded.view_weights(canonical_directions(["front"])).shape == (1, 3)


def direct_set():
    """Hand-built cache with non-collinear 2D embeddings (T=1, D=2).

    Concept 0: front=[1,0], side=[0,1], top=[0,0.5]; concept 1 mirrors it.
    uncond = [0,0]. Non-collinear so perp components are non-trivial.
    """
    embeds = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]],  # concept 0
            [[2.0, 0.0], [0.0, 2.0], [0.0, 1.0]],  # concept 1
        ]
    )[None].reshape(2, 3, 1, 2)
    return ViewPromptSet(
        embeds=embeds,
        static_embeds=torch.zeros(1, 1, 2),
        uncond=torch.zeros(1, 2),
        texts=["f", "s", "t", "f", "s", "t"],
        static_texts=[""],
        views=list(VIEWS),
    )


def test_perp_orthogonal_to_positive_multitoken():
    # Invariant test with real multi-token shapes (T=2): every perp must
    # be orthogonal to the positive direction. Catches partial flattening
    # bugs that single-token fixtures hide.
    vs = build()
    torch.manual_seed(0)
    d = torch.randn(4, len(VIEWS), T, D)
    d0 = torch.randn(4, T, D)
    perp = vs._perp(d, d0)
    assert perp.shape == d.shape
    # Orthogonality holds over the flattened (T, D) score vector — per-
    # token dots cancel pairwise, so assert the global dot.
    dots = (perp * d0[:, None]).sum(-1).sum(-1)  # (B, V)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-3)


def test_perpneg_plan():
    vs = build()
    front = canonical_directions(["front"])
    pos, neg = vs.perpneg_plan(front, weighting="fixed", scale=1.5)
    assert pos.item() == 0
    assert torch.allclose(neg, torch.tensor([[0.0, -1.5, -1.5]]))
    pos, neg = vs.perpneg_plan(front, weighting="sphere", scale=1.0)
    w = vs.view_weights(front)[0]
    expected = -1.0 * (1.0 - w)
    expected[0] = 0.0
    assert torch.allclose(neg[0], expected)
    with pytest.raises(ValueError):
        vs.perpneg_plan(front, weighting="bogus")


def test_perpneg_embeds_exact_front():
    vs = direct_set()
    front = canonical_directions(["front"])
    # Concept 0 only, so the hand-computed numbers hold.
    composite = vs.perpneg_embeds(
        front,
        concept_weights=torch.tensor([[1.0, 0.0]]),
        weighting="fixed",
        scale=1.0,
    )
    # d_front = [1,0]; perp(d_side) = [0,1]; perp(d_top) = [0,0.5];
    # composite = E_front - 1*perp_side - 1*perp_top + U.
    assert torch.allclose(composite[0], torch.tensor([1.0, -1.5]))


def test_perpneg_embeds_parallel_negative_vanishes():
    vs = direct_set()
    # Concept-mixed view directions are all collinear for this camera:
    # front-direction only → perp of everything parallel to it is ~0, so
    # the composite equals the positive embedding itself.
    composite = vs.perpneg_embeds(
        canonical_directions(["front"]),
        concept_weights=torch.tensor([[1.0, 0.0]]),
        weighting="fixed",
        scale=5.0,
    )
    # Still not zero here (side/top are orthogonal in embedding space),
    # but a purely parallel negative must leave the composite unchanged:
    collinear = ViewPromptSet(
        embeds=torch.tensor([[[3.0, 0.0], [1.5, 0.0]]])[None].reshape(1, 2, 1, 2),
        static_embeds=torch.zeros(1, 1, 2),
        uncond=torch.zeros(1, 2),
        texts=["f", "s"],
        static_texts=[""],
        views=["front", "side"],
    )
    out = collinear.perpneg_embeds(
        canonical_directions(["front"]),
        weighting="fixed",
        scale=10.0,
    )
    assert torch.allclose(out[0], torch.tensor([3.0, 0.0]))


def test_perpneg_combine_math():
    # V=2 branches, B=2 renders, 3-channel scores.
    eps_branches = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # branch 0
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],  # branch 1
        ]
    )
    eps_uncond = torch.zeros(2, 3)
    pos_idx = torch.tensor([0, 1])  # render 1's positive is branch 1
    neg_weights = torch.tensor([[0.0, -1.0], [-1.0, 0.0]])
    out = perpneg_combine(eps_branches, eps_uncond, pos_idx, neg_weights, 2.0)
    # Both renders: positive direction [1,0,0] at scale 2, negative
    # [0,1,0] fully orthogonal -> subtracted at weight -1.
    assert torch.allclose(out, torch.tensor([[2.0, -1.0, 0.0], [2.0, -1.0, 0.0]]))


def test_perpneg_combine_parallel_negative():
    eps_branches = torch.tensor([[[2.0, 0.0, 0.0]]])  # parallel to positive
    eps_uncond = torch.zeros(1, 3)
    out = perpneg_combine(
        eps_branches,
        eps_uncond,
        torch.tensor([0]),
        torch.tensor([[0.0, -1.0]]),
        1.0,
    )
    # Parallel negative -> perp = 0 -> plain CFG result.
    assert torch.allclose(out, torch.tensor([[2.0, 0.0, 0.0]]))


def test_bsd_perpneg_branches_and_layout():
    vs = direct_set()
    front = canonical_directions(["front"])
    branches = vs.bsd_perpneg_branches(front, weighting="fixed", scale=1.0)
    assert branches.shape == (2, 1, 1, 2)
    # Concept 0: E_front - perp(side) - perp(top) + U = [1, -1.5].
    assert torch.allclose(branches[0, 0, 0], torch.tensor([1.0, -1.5]))
    # Concept 1: d = 2x concept 0's -> same directions, same perp -> [2, -3].
    assert torch.allclose(branches[1, 0, 0], torch.tensor([2.0, -3.0]))
    # Feeding them through bsd_embeds keeps the BSD row layout intact.
    attn_alphas = torch.tensor([[0.6, 0.4]])  # 2 concepts + blank(0 cols extra)
    vs_static = direct_set()
    embeds = vs_static.bsd_embeds(
        front, attn_alphas, num_funcs=2, branches=branches
    )
    # Rows: [c0 x B, c1 x B, blank x B, uncond x B].
    assert torch.allclose(embeds[0, 0], branches[0, 0, 0])
    assert torch.allclose(embeds[1, 0], branches[1, 0, 0])
    assert torch.allclose(embeds[2, 0], torch.zeros(1, 2))  # blank branch
    assert torch.allclose(embeds[3, 0], torch.zeros(1, 2))  # uncond (U = 0)


def test_branch_embeds_concept_mix():
    vs = build()
    branches = vs.branch_embeds(canonical_directions(["front"]))
    assert branches.shape == (len(VIEWS), 1, T, D)
    # Uniform concept mix: branch v = 0.5*(fill v) + 0.5*(fill V+v) = v + V/2.
    assert torch.allclose(
        branches[:, 0, 0, 0],
        torch.arange(len(VIEWS), dtype=torch.float32) + len(VIEWS) / 2,
    )
