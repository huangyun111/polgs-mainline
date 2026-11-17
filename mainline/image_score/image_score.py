"""RDGS-style detached image error backprojection through native color compositing."""
from types import SimpleNamespace
import time
import torch
import torch.nn.functional as F
from density_state import fingerprint,model_state,rng_state

WINDOW=None
def error_map(pred,obs,raw=False):
    global WINDOW
    from utils.loss_utils import create_window
    x,y=pred.detach().unsqueeze(0),obs.detach().unsqueeze(0)
    assert x.shape==y.shape and x.shape[1]==3
    if WINDOW is None or WINDOW.device!=x.device:WINDOW=create_window(11,3).to(device=x.device,dtype=x.dtype)
    w=WINDOW;m1=F.conv2d(x,w,padding=5,groups=3);m2=F.conv2d(y,w,padding=5,groups=3)
    v1=F.conv2d(x*x,w,padding=5,groups=3)-m1*m1;v2=F.conv2d(y*y,w,padding=5,groups=3)-m2*m2
    cov=F.conv2d(x*y,w,padding=5,groups=3)-m1*m2
    s=((2*m1*m2+.01**2)*(2*cov+.03**2))/((m1*m1+m2*m2+.01**2)*(v1+v2+.03**2))
    value=((1-s.mean(1)[0])*.5).detach()
    return value if raw else value.clamp_min(0)

def readonly(g,cam,subset=None,overlap=False):
    ids=slice(None) if subset is None else subset
    xyz=g.get_xyz.detach()[ids];scales=g.get_scaling.detach()[ids];rot=g.get_rotation.detach()[ids];op=g.get_opacity.detach()[ids]
    if overlap:
        xyz=xyz[:1].expand(3,-1).contiguous();scales=scales[:1].expand(3,-1).contiguous();rot=rot[:1].expand(3,-1).contiguous();op=torch.tensor([[.2],[.4],[.7]],device=xyz.device,dtype=xyz.dtype)
    pc=SimpleNamespace(get_xyz=xyz,get_scaling=scales,get_rotation=rot,get_opacity=op,active_sh_degree=g.active_sh_degree,config=g.config)
    freeze=lambda x:x.detach() if torch.is_tensor(x) else x
    c=SimpleNamespace(**{k:freeze(getattr(cam,k)) for k in ['FoVx','FoVy','image_height','image_width','world_view_transform','full_proj_transform','prcppoint','camera_center']})
    patch=torch.tensor([0,0,c.image_height,c.image_width],device=xyz.device,dtype=torch.float32)
    c.to_device=lambda:None;c.update=lambda:None;c.random_patch=lambda h,w:patch
    return pc,c

def backproject(native,pc,cam,pipe,err):
    with torch.enable_grad():
        scalar=torch.zeros((len(pc.get_xyz),1),device=pc.get_xyz.device,dtype=pc.get_xyz.dtype,requires_grad=True)
        pkg=native(cam,pc,pipe,torch.zeros(3,device=scalar.device),[float('inf'),float('inf')],override_color=scalar.expand(-1,3).contiguous(),initial_stage=True)
        score=torch.autograd.grad((pkg['render'][0]*err.detach()).sum(),scalar)[0].detach().flatten()
    assert bool(torch.isfinite(score).all()) and bool((score>=-1e-6).all())
    return score.clamp_min(0)

def fixture(native,g,cam,pipe,visible):
    ids=torch.nonzero(visible,as_tuple=False).flatten()[:1];assert len(ids)==1
    pc,c=readonly(g,cam,ids,overlap=True)
    err=torch.linspace(.1,1.,c.image_height*c.image_width,device=pc.get_xyz.device).reshape(c.image_height,c.image_width)
    actual=backproject(native,pc,c,pipe,err);expected=[]
    with torch.no_grad():
        for i in range(3):
            colors=torch.zeros((3,3),device=pc.get_xyz.device);colors[i,0]=1
            image=native(c,pc,pipe,torch.zeros(3,device=colors.device),[float('inf'),float('inf')],override_color=colors,initial_stage=True)['render'][0]
            expected.append((image*err).sum())
    expected=torch.stack(expected);assert bool((expected>0).all())
    assert torch.allclose(actual,expected,atol=1e-5,rtol=1e-4),(actual,expected)
    zero=backproject(native,pc,c,pipe,torch.zeros_like(err));assert bool((zero==0).all())
    return dict(passed=True,overlapping_Gaussians=3,VJP=actual.cpu().tolist(),independent_onehot_integrals=expected.cpu().tolist(),zero_error_gives_zero=True,atol=1e-5,rtol=1e-4)

def accumulate(g,native,cam,pipe,pred,obs,visible,depth_cache,state,deep=False):
    begin=time.monotonic();ps=[p for group in g.optimizer.param_groups for p in group['params']]
    versions=[p._version for p in ps];grads=[p.grad for p in ps]
    if deep:before_model=fingerprint(model_state(g));before_rng=fingerprint(rng_state());before_depth=fingerprint(depth_cache)
    ev=[torch.cuda.Event(enable_timing=True) for _ in range(3)];ev[0].record();err=error_map(pred,obs);ev[1].record()
    pc,c=readonly(g,cam);score=backproject(native,pc,c,pipe,err);ev[2].record();ev[2].synchronize()
    assert len(score)==len(g._image_score);g._image_score=torch.maximum(g._image_score,score)
    g._image_views.add(int(cam.uid));state['image_score_calls']+=1
    state['image_error_gpu_ms']+=ev[0].elapsed_time(ev[1]);state['image_backproject_gpu_ms']+=ev[1].elapsed_time(ev[2])
    state['image_score_wall_seconds']+=time.monotonic()-begin
    assert versions==[p._version for p in ps] and all(p.grad is grad for p,grad in zip(ps,grads))
    if deep:
        from utils.loss_utils import ssim
        raw=error_map(pred,obs,raw=True);native_dssim=(1-ssim(pred.detach(),obs.detach()))*.5
        scalar_audit=dict(raw_DSSIM_mean=float(raw.mean()),native_DSSIM=float(native_dssim),clipped_DSSIM_mean=float(err.mean()),nonnegative_clip_mean_shift=float((err-raw).mean()))
        assert abs(float(raw.mean()-native_dssim))<1e-6,scalar_audit
        audit=fixture(native,g,cam,pipe,visible)
        assert before_model==fingerprint(model_state(g)) and before_rng==fingerprint(rng_state()) and before_depth==fingerprint(depth_cache)
        return dict(passed=True,model_Adam_statistics_and_gradients_unchanged=True,RNG_and_depth_unchanged=True,DSSIM_matches_native_SSIM_before_nonnegative_clip=True,DSSIM_scalar_audit=scalar_audit,error_mean=float(err.mean()),native_color_contribution_fixture=audit)
    return None
