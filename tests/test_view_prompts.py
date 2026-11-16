import pytest
import torch
from types import SimpleNamespace

from vdsds.utils.view_prompts import (
    ViewPromptSet,
    canonical_directions,
    format_view_weights,
)

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
