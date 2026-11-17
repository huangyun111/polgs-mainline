"""Independent budget control. No peer exchange, quota files, or GT access."""
import json
import numpy as np
import torch
from routing_core import STATS

CAP=39000

def growth(c,s,room):
    assert min(c,s,room)>=0
    if c+s<=room:return c,s
    kc=room*c//(c+s)
    return kc,room-kc

def rank(score,mask,k):
    ids=torch.nonzero(mask,as_tuple=False).flatten().cpu().numpy()
    values=score.detach().cpu().numpy()[ids]
    return torch.as_tensor(ids[np.lexsort((ids,-values))[:k]],device=score.device,dtype=torch.long)

def pools(g,stats,threshold,extent):
    def avg(key):
        v=stats[key]/stats['denom'];v[v.isnan()]=0
        return v[:,0]
    score=avg('xyz_gradient_accum');scale=avg('scale_gradient_accum');opacity=avg('opac_gradient_accum')
    small=g.get_scaling.max(1).values<=g.percent_dense*extent
    return score,small&(scale<=1e-7)&(opacity<=2)&(score>=threshold),(~small)&(score>=threshold)

def common_add(g,ref,scale,visible):
    bank=g._common_stats
    bank['xyz_gradient_accum'][visible]+=torch.norm(ref[visible,:2],dim=-1,keepdim=True)
    bank['scale_gradient_accum'][visible]+=scale[visible,:2].sum(1,True)
    bank['rot_gradient_accum'][visible]+=torch.norm(g._rotation[visible],dim=-1,keepdim=True)
    bank['opac_gradient_accum'][visible]+=g._opacity[visible]
    bank['denom'][visible]+=1

def install(g,W,R,arm,state,opt):
    assert arm in ['N-cap','S-cap','S-dual']
    dual=arm=='S-dual';native_prune=g.adaptive_prune;native_points=g.prune_points
    native_clone=g.densify_and_clone;native_split=g.densify_and_split
    if dual:g._common_stats={k:getattr(g,k).detach().clone() for k in STATS}
    state.update(Gaussian_forward_steps_new=0,Gaussian_Adam_exposure_new=0,budget_operations=0,clone_births=0,split_births=0,split_parents_removed=0,clone_births_used=0,split_births_used=0,quota_extra_pruned=0,native_pruned=0,capacity_growth_limited_operations=0,opacity_cap_operations=0)
    pending={};birth_phase=[False]
    def prune_points(mask):
        if dual and not birth_phase[0]:
            assert all(len(v)==len(mask) for v in g._common_stats.values())
            g._common_stats={k:v[~mask] for k,v in g._common_stats.items()}
        native_points(mask)
    g.prune_points=prune_points
    def prune(min_opacity,extent):
        pre=len(g._xyz);native_prune(min_opacity,extent);native_n=len(g._xyz)
        keep=min(native_n,CAP);assert keep>0
        extra=native_n-keep
        if extra:
            opacity=g.get_opacity.detach().flatten()
            ids=rank(opacity,torch.ones(native_n,device=opacity.device,dtype=torch.bool),keep)
            mask=torch.ones(native_n,device=opacity.device,dtype=torch.bool);mask[ids]=False;g.prune_points(mask)
        assert len(g._xyz)==keep
        if dual:
            assert all(len(v)==keep for v in g._common_stats.values())
            for k in ['denom','rot_gradient_accum','opac_gradient_accum']:assert torch.equal(g._common_stats[k],getattr(g,k))
        pending.update(before=pre,native_remaining=native_n,retained=keep,extra_pruned=extra)
        state['native_pruned']+=pre-native_n;state['quota_extra_pruned']+=extra;state['opacity_cap_operations']+=int(extra>0)
    def densify(threshold,extent):
        assert threshold==opt.densify_grad_threshold and len(g._xyz)==pending['retained']
        primary={k:getattr(g,k) for k in STATS}
        score,clone,split=pools(g,primary,threshold,extent)
        nc,ns=int(clone.sum()),int(split.sum());c,s=nc,ns
        cc=cs=None
        if dual:
            _,common_clone,common_split=pools(g,g._common_stats,threshold,extent)
            cc,cs=int(common_clone.sum()),int(common_split.sum());c,s=min(c,cc),min(s,cs)
        room=CAP-len(g._xyz);kc,ks=growth(c,s,room)
        ci=rank(score,clone,kc);si=rank(score,split,ks)
        assert len(ci)==kc and len(si)==ks and bool(clone[ci].all()) and bool(split[si].all())
        n=len(g._xyz);birth_phase[0]=True
        try:
            fake=torch.zeros((n,1),device=g._xyz.device);fake[ci]=1
            native_clone(fake,.5,extent,pre_mask=torch.ones(n,device=g._xyz.device,dtype=torch.bool))
            fake=torch.zeros((n,1),device=g._xyz.device);fake[si]=1
            native_split(fake,.5,extent)
        finally:birth_phase[0]=False
        after=len(g._xyz);assert after==n+kc+ks and after<=CAP
        assert all(bool((getattr(g,k)==0).all()) for k in STATS)
        if dual:g._common_stats={k:torch.zeros_like(getattr(g,k)) for k in STATS}
        limited=c+s>room
        state['budget_operations']+=1;state['capacity_growth_limited_operations']+=int(limited)
        state['clone_births']+=kc;state['split_births']+=2*ks;state['split_parents_removed']+=ks
        if state['iteration']<15000:state['clone_births_used']+=kc;state['split_births_used']+=2*ks
        journal=dict(pending,iteration=state['iteration'],arm=arm,clone_candidates=nc,split_candidates=ns,common_clone_candidates=cc,common_split_candidates=cs,demand_clone=c,demand_split=s,Kc=kc,Ks=ks,after=after,cap=CAP,growth_limited=limited,used_in_final_PLY=state['iteration']<15000)
        with (W/'budget_operations.jsonl').open('a') as f:f.write(json.dumps(journal)+'\n')
        parents=W/'parents';parents.mkdir(exist_ok=True)
        np.savez_compressed(parents/(str(state['iteration'])+'.npz'),clone=ci.cpu().numpy(),split=si.cpu().numpy())
        state.update(budget_last_retained=n,budget_last_Kc=kc,budget_last_Ks=ks,budget_cap=CAP)
    g.adaptive_prune=prune;g.adaptive_densify=densify
