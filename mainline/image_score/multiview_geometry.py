"""PGSR geometric term adapted to unchanged PolGS camera-z and cameras."""
import numpy as np
import torch
import torch.nn.functional as F

def build_pairs(views):
    C=torch.stack([c.camera_center for c in views]).detach().cpu().double().numpy()
    axis=np.array([c.world_view_transform[:3,2].detach().cpu().numpy() for c in views])
    axis=axis/np.linalg.norm(axis,axis=1,keepdims=True)
    dist=np.linalg.norm(C[:,None]-C[None],axis=-1)
    angle=np.degrees(np.arccos(np.clip(axis@axis.T,-1,1)))
    pairs=[]
    for i in range(len(views)):
        order=np.lexsort((angle[i],dist[i]));pairs.append([int(j) for j in order if angle[i,j]<30 and .01<dist[i,j]<1.5][:8])
    return pairs,dict(distance=dist.tolist(),angle=angle.tolist(),pairs=pairs)

def calibration(cam,like):
    K=torch.as_tensor(cam.HWK[2][:3,:3],device=like.device,dtype=like.dtype)
    return K,cam.world_view_transform.to(device=like.device,dtype=like.dtype)

def geometry_loss(zref,zsrc,ref,src,*,frozen_mask=None,frozen_weights=None):
    zref=zref.squeeze();zsrc=zsrc.squeeze();H,W=zref.shape
    K,T=calibration(ref,zref);Ks,Ts=calibration(src,zref)
    y,x=torch.meshgrid(torch.arange(H,device=zref.device,dtype=zref.dtype),torch.arange(W,device=zref.device,dtype=zref.dtype),indexing='ij')
    pixels=torch.stack((x,y),-1).reshape(-1,2)
    rays=torch.cat((pixels,torch.ones_like(pixels[:,:1])),-1)@torch.linalg.inv(K).T
    ref_valid=torch.isfinite(zref.reshape(-1))&(zref.reshape(-1)>.1)
    depth=torch.where(ref_valid,zref.reshape(-1),torch.ones_like(zref.reshape(-1)))
    p=rays*depth[:,None]
    world=(p-T[3,:3])@T[:3,:3].T
    ps=world@Ts[:3,:3]+Ts[3,:3]
    valid_z=torch.isfinite(ps).all(-1)&(ps[:,2]>.1)
    safe_z=torch.where(valid_z,ps[:,2],torch.ones_like(ps[:,2]))
    projected=ps@Ks.T;uv=projected[:,:2]/safe_z[:,None]
    hs,ws=zsrc.shape
    inside=(uv[:,0]>0)&(uv[:,0]<ws)&(uv[:,1]>0)&(uv[:,1]<hs)
    raw_grid=2*uv/uv.new_tensor([ws-1,hs-1])-1
    grid=torch.where((ref_valid&valid_z&torch.isfinite(raw_grid).all(-1))[:,None],raw_grid,torch.zeros_like(raw_grid))
    finite_source=torch.isfinite(zsrc)&(zsrc>.1)
    source_clean=torch.where(finite_source,zsrc,torch.zeros_like(zsrc))
    sample=F.grid_sample(source_clean[None,None],grid.reshape(1,-1,1,2),mode='bilinear',padding_mode='border',align_corners=True).reshape(-1)
    # Invalid source samples contribute neither a zero-depth point nor NaN gradients.
    validity=F.grid_sample(finite_source.to(zsrc.dtype)[None,None],grid.detach().reshape(1,-1,1,2),mode='bilinear',padding_mode='border',align_corners=True).reshape(-1)
    sampled_valid=validity>1-1e-6
    back_s=ps/safe_z[:,None]*sample[:,None]
    back_world=(back_s-Ts[3,:3])@Ts[:3,:3].T
    back=back_world@T[:3,:3]+T[3,:3]
    valid_back=torch.isfinite(back).all(-1)&(back[:,2]>.1)
    bz=torch.where(valid_back,back[:,2],torch.ones_like(back[:,2]))
    back_h=back@K.T;returned=back_h[:,:2]/bz[:,None]
    error=torch.linalg.norm(returned-pixels,dim=-1)
    valid=ref_valid&valid_z&inside&sampled_valid&valid_back&torch.isfinite(error)
    mask=valid&(error<1.) if frozen_mask is None else frozen_mask
    weights=torch.exp(-error).detach() if frozen_weights is None else frozen_weights
    loss=.03*(weights[mask]*error[mask]).mean() if bool(mask.any()) else (torch.where(ref_valid.reshape(H,W),zref,torch.zeros_like(zref)).sum()+source_clean.sum())*0.
    meta=dict(error=error,valid=valid,mask=mask,weights=weights,grid=grid,sampled_depth=sample,
              pixels=pixels,returned=returned,shape=(H,W))
    return loss,meta

def support_at_anchors(meta,cam,path):
    # Same fixed232 anchors; no GT arrays. Diagnostic does not alter loss support.
    data=np.load(path,allow_pickle=False);original_ids=[0,1,2,4,5,7,8,9,11,12,14,15,16,18]
    original=original_ids[int(cam.image_name)];rows=[]
    H,W=meta['shape']
    with torch.no_grad():
        for roi in ('ROI0','ROI1'):
            ids=np.flatnonzero(np.array([int(v)==original for v in data[roi+'_view_ids']]))
            if not len(ids):continue
            points=torch.as_tensor(data[roi+'_O'][ids]+data[roi+'_R'][ids],device=cam.world_view_transform.device,dtype=cam.world_view_transform.dtype)
            clip=torch.cat((points,torch.ones_like(points[:,:1])),-1)@cam.full_proj_transform
            uv=(clip[:,:2]/clip[:,3:]+1)*points.new_tensor([W,H])*.5-.5
            xy=uv.round().long();valid=(xy[:,0]>=0)&(xy[:,0]<W)&(xy[:,1]>=0)&(xy[:,1]<H)
            for j,i in enumerate(ids):
                if not bool(valid[j]):rows.append(dict(ROI=roi,ray=int(i),valid=False,kept=False,error=None));continue
                k=int(xy[j,1])*W+int(xy[j,0]);e=float(meta['error'][k])
                rows.append(dict(ROI=roi,ray=int(i),valid=bool(meta['valid'][k]),kept=bool(meta['mask'][k]),error=e if np.isfinite(e) else None))
    return rows
