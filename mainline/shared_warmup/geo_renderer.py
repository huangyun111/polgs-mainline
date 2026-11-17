"""Share ONE native forward; preserve old VJP and separate new depth-loss VJP."""
from pathlib import Path
import torch
import diff_gaussian_rasterization as native
from extension_loader import extension as corrected
from slope_proxy import native_slopes

COUNTS=dict(forward=0,common_backward=0,geometry_backward=0)

class DualDepth(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means3D,means2D,sh,colors,opacity,scales,rotations,cov,view,proj,campos,slopes,settings):
        args=(settings.bg,means3D,colors,opacity,scales,rotations,settings.scale_modifier,cov,view,proj,
              settings.prcppoint,settings.patch_bbox,settings.tanfovx,settings.tanfovy,settings.image_height,
              settings.image_width,sh,settings.sh_degree,campos,settings.prefiltered,settings.debug,settings.config)
        num,color,normal,depth,opac,radii,geom,binning,image=native._C.rasterize_gaussians(*args)
        ctx.settings=settings;ctx.num=num
        ctx.save_for_backward(colors,means3D,scales,rotations,cov,radii,sh,geom,binning,image)
        ctx.set_materialize_grads(False)
        COUNTS['forward']+=1
        return color,normal,depth,opac,radii,depth.clone()

    @staticmethod
    def backward(ctx,gc,gn,gd,go,gr,gg):
        colors,means,scales,rot,cov,radii,sh,geom,binning,image=ctx.saved_tensors;s=ctx.settings
        H,W=s.image_height,s.image_width
        zeros=lambda n:torch.zeros((n,H,W),device=means.device,dtype=means.dtype)
        def args(c,n,d,o):
            return (s.bg,means,radii,colors,scales,rot,s.scale_modifier,cov,s.viewmatrix,s.projmatrix,s.prcppoint,
                    s.patch_bbox,s.tanfovx,s.tanfovy,c,n,d,o,sh,s.sh_degree,s.campos,geom,ctx.num,binning,image,s.debug,s.config)
        base=None
        if any(t is not None for t in (gc,gn,gd,go)):
            base=native._C.rasterize_gaussians_backward(*args(gc if gc is not None else zeros(3),gn if gn is not None else zeros(3),gd if gd is not None else zeros(1),go if go is not None else zeros(1)))
            COUNTS['common_backward']+=1
        slopes=None
        if gg is not None:
            new=corrected.rasterize_gaussians_backward(*args(zeros(3),zeros(3),gg,zeros(1)))
            assert len(new)==12
            base=new[:11] if base is None else tuple(a+b for a,b in zip(base,new[:11]))
            slopes=new[11];COUNTS['geometry_backward']+=1
        if base is None:return (None,)*13
        gm2,c,op,gm3,gcov,gsh,gs,gro,gv,gp,gcam=base
        return gm3,gm2,gsh,c,op,gs,gro,gcov,gv,gp,gcam,slopes,None

class GaussianRasterizer(native.GaussianRasterizer):
    def forward(self,means3D,means2D,opacities,shs=None,colors_precomp=None,scales=None,rotations=None,cov3D_precomp=None):
        empty=lambda:torch.empty(0,device=means3D.device,dtype=means3D.dtype)
        assert scales is not None and rotations is not None and cov3D_precomp is None
        s=self.raster_settings
        return DualDepth.apply(means3D,means2D,shs if shs is not None else empty(),colors_precomp if colors_precomp is not None else empty(),
                               opacities,scales,rotations,empty(),s.viewmatrix,s.projmatrix,s.campos,
                               native_slopes(means3D,rotations,s.viewmatrix,s),s)

def make_renderer(gr):
    path=Path(__file__).resolve().parent
    source=Path(gr.__file__).read_text()
    old='from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer'
    assert source.count(old)==1
    modified=source.replace(old,'from diff_gaussian_rasterization import GaussianRasterizationSettings\nfrom geo_renderer import GaussianRasterizer')
    old='rendered_image, rendered_normal, rendered_depth, rendered_opac, radii = rasterizer('
    assert modified.count(old)==1
    modified=modified.replace(old,'rendered_image, rendered_normal, rendered_depth, rendered_opac, radii, geo_depth = rasterizer(')
    modified=modified.replace('return {"render": rendered_image,','return {"geo_depth": geo_depth, "render": rendered_image,')
    (path/'renderer_original.py').write_text(source);(path/'renderer_geo.py').write_text(modified)
    scope=dict(gr.__dict__);exec(compile(modified,str(path/'renderer_geo.py'),'exec'),scope)
    return scope['render']
