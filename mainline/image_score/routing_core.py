"""Exact common and native-joint gradients; no subtractive gradient approximation."""
import torch

def clone(x):return None if x is None else x.detach().clone()
def clear_screens(ref,src):ref.grad=None;src.grad=None
def compare(a,b):
    if a is None or b is None:return dict(passed=a is None and b is None,max_abs=0.,relative_l2=0.)
    if a.numel()==0:return dict(passed=tuple(a.shape)==tuple(b.shape),max_abs=0.,relative_l2=0.)
    delta=(a-b).detach();return dict(passed=bool(torch.allclose(a,b,atol=1e-6,rtol=1e-5)),max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/b.detach().norm().clamp_min(1e-20)))

def gradients(common_loss,joint_loss,params,ref_screen,src_screen,oracle=False):
    pre_nonnull=[p.grad is not None for p in params]
    if oracle:
        assert all(p.grad is None or bool((p.grad==0).all()) for p in params),'Native zero_grad/prefix must leave zero or None'
    ca,cb,ja,jb=[torch.cuda.Event(enable_timing=True) for _ in range(4)];ca.record()
    raw=torch.autograd.grad(common_loss,params+[ref_screen,src_screen],retain_graph=True,allow_unused=True)
    cb.record()
    common=[clone(g) if g is not None else torch.zeros_like(p) if existed else None for p,g,existed in zip(params,raw[:-2],pre_nonnull)]
    cref=clone(raw[-2]);csrc=clone(raw[-1]);assert cref is not None
    clear_screens(ref_screen,src_screen)
    if oracle:assert all(p.grad is None or bool((p.grad==0).all()) for p in params),'autograd.grad must not contaminate leaf .grad'
    ja.record();joint_loss.backward(retain_graph=oracle);jb.record();jb.synchronize()
    joint=[p.grad for p in params];jref=ref_screen.grad;jsrc=src_screen.grad;assert jref is not None
    audit=None
    if oracle:
        joint=[clone(g) for g in joint];jref=clone(jref);jsrc=clone(jsrc)
        for p,existed in zip(params,pre_nonnull):p.grad=torch.zeros_like(p) if existed else None
        clear_screens(ref_screen,src_screen);common_loss.backward()
        checks=[compare(a,p.grad) for a,p in zip(common,params)];checks.append(compare(cref,ref_screen.grad));checks.append(compare(csrc,src_screen.grad))
        assert all(x['passed'] for x in checks),checks
        for p,g in zip(params,joint):p.grad=g
        ref_screen.grad=jref;src_screen.grad=jsrc
        audit=dict(passed=True,common_autograd_matches_native_common_backward=True,atol=1e-6,rtol=1e-5,checks=checks,pre_gradient_none_flags_preserved=True,retained_screens_cleared_between_passes=True)
    return dict(common_gradient_ms=ca.elapsed_time(cb),joint_backward_ms=ja.elapsed_time(jb),common=common,joint=joint,common_ref=cref,joint_ref=jref,common_src=csrc,joint_src=jsrc,audit=audit)

def assign_optimizer(params,result,use_geometry):
    chosen=result['joint'] if use_geometry else result['common']
    for p,g in zip(params,chosen):p.grad=g
    return chosen

STATS=('xyz_gradient_accum','scale_gradient_accum','rot_gradient_accum','opac_gradient_accum','denom')
def statistics_fixture(g,original,ref,visibility,common_ref,joint_ref,common_scale,joint_scale):
    """Call the original stats at the real state twice, restore arrays exactly."""
    pre={k:getattr(g,k).detach().clone() for k in STATS};saved_ref=ref.grad;saved_scale=g._scaling.grad;expected={};checks={}
    try:
        with torch.no_grad():
            for name,rg,sg in [('common',common_ref,common_scale),('joint',joint_ref,joint_scale)]:
                assert rg is not None and sg is not None
                for k in STATS:getattr(g,k).copy_(pre[k])
                want={k:v.clone() for k,v in pre.items()}
                want['xyz_gradient_accum'][visibility]+=torch.norm(rg[visibility,:2],dim=-1,keepdim=True)
                want['scale_gradient_accum'][visibility]+=sg[visibility,:2].sum(1,True)
                want['rot_gradient_accum'][visibility]+=torch.norm(g._rotation[visibility],dim=-1,keepdim=True)
                want['opac_gradient_accum'][visibility]+=g._opacity[visibility]
                want['denom'][visibility]+=1
                ref.grad=rg;g._scaling.grad=sg;original(ref,visibility)
                checks[name]={k:bool(torch.equal(getattr(g,k),want[k])) for k in STATS};assert all(checks[name].values()),checks
                expected[name]=want
    finally:
        with torch.no_grad():
            for k in STATS:getattr(g,k).copy_(pre[k])
        ref.grad=saved_ref;g._scaling.grad=saved_scale
    separation={k:float((expected['joint'][k]-expected['common'][k]).norm()) for k in STATS}
    return dict(passed=True,native_statistics_match_independent_formulas=checks,common_joint_statistics_difference_norm=separation,no_extra_source_screen_statistics=True),expected
