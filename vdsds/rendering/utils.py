import nvdiffrast.torch as dr
from jaxtyping import Float
from torch import Tensor

from .context import RenderContext


def attr_map(ctx: RenderContext, val: Float[Tensor, "..."]) -> Float[Tensor, "B H W C"]:
    mesh = ctx.mesh
    if len(val.shape) == 1:
        val = val[None]
    if len(val.shape) == 2:
        if val.shape[0] == 1:
            val = val.expand(mesh.num_V, 3)
        return dr.interpolate(val.contiguous(), ctx.rast, mesh.F)[0]
    if len(val.shape) == 3:
        val = val[None]
    # Assume B C H W format for texture
    return dr.texture(
        val.permute(0, 2, 3, 1).contiguous(),
        attr_map(ctx, mesh.uv_co),
        filter_mode=ctx.get("filter_mode", "linear"),
        boundary_mode=ctx.get("boundary_mode", "wrap"),
    )
