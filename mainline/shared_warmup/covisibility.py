"""Pair selection only. Coordinates/thresholds are in original native pixels."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def directional(zref, zsrc, mref, msrc, ref, src, stride=4):
    zref,zsrc,mref,msrc=[x.squeeze() for x in (zref,zsrc,mref,msrc)]
    H,W=zref.shape; hs,ws=zsrc.shape
    K=torch.as_tensor(ref.HWK[2][:3,:3],device=zref.device,dtype=zref.dtype)
    Ks=torch.as_tensor(src.HWK[2][:3,:3],device=zref.device,dtype=zref.dtype)
    T=ref.world_view_transform.to(zref); Ts=src.world_view_transform.to(zref)
    y,x=torch.meshgrid(torch.arange(0,H,stride,device=zref.device,dtype=zref.dtype),torch.arange(0,W,stride,device=zref.device,dtype=zref.dtype),indexing='ij')
    pixels=torch.stack((x,y),-1).reshape(-1,2)
    zr=zref[::stride,::stride].reshape(-1)
    valid_ref=torch.isfinite(zr)&(zr>.1)&(mref[::stride,::stride].reshape(-1)>.5)
    rays=torch.cat((pixels,torch.ones_like(pixels[:,:1])),-1)@torch.linalg.inv(K).T
    p=rays*torch.where(valid_ref,zr,torch.ones_like(zr))[:,None]
    world=(p-T[3,:3])@T[:3,:3].T
    ps=world@Ts[:3,:3]+Ts[3,:3]
    good=torch.isfinite(ps).all(-1)&(ps[:,2]>.1)
    sz=torch.where(good,ps[:,2],torch.ones_like(ps[:,2]))
    uv=(ps@Ks.T)[:,:2]/sz[:,None]
    inside=(uv[:,0]>0)&(uv[:,0]<ws-1)&(uv[:,1]>0)&(uv[:,1]<hs-1)
    grid=2*uv/uv.new_tensor([ws-1,hs-1])-1
    grid=torch.where(torch.isfinite(grid),grid,torch.zeros_like(grid)).reshape(1,-1,1,2)
    source_valid=torch.isfinite(zsrc)&(zsrc>.1)&(msrc>.5)
    def query(a):
        return F.grid_sample(a[None,None],grid,mode='bilinear',padding_mode='border',align_corners=True).reshape(-1)
    sampled=query(torch.where(source_valid,zsrc,torch.zeros_like(zsrc)))
    full_support=query(source_valid.to(zref.dtype))>1-1e-6
    consistent=(sampled-sz).abs()<=.01*sz
    back_s=ps/sz[:,None]*sampled[:,None]
    back_world=(back_s-Ts[3,:3])@Ts[:3,:3].T
    back=back_world@T[:3,:3]+T[3,:3]
    back_good=torch.isfinite(back).all(-1)&(back[:,2]>.1)
    bz=torch.where(back_good,back[:,2],torch.ones_like(back[:,2]))
    returned=(back@K.T)[:,:2]/bz[:,None]
    error=torch.linalg.norm(returned-pixels,dim=-1)
    mask=valid_ref&good&inside&full_support&consistent&back_good&torch.isfinite(error)&(error<1.)
    count=int(mask.sum());denom=int(valid_ref.sum())
    return dict(supported=count,reference_valid_foreground=denom,fraction=count/max(denom,1))


@torch.no_grad()
def build_covisibility(views,depths,masks):
    n=len(views);rows=[[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i!=j:rows[i][j]=directional(depths[i],depths[j],masks[i],masks[j],views[i],views[j])
    scores=[[0.]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j:continue
            a,b=rows[i][j],rows[j][i]
            if min(a['supported'],b['supported'])>=64:
                x,y=a['fraction'],b['fraction'];scores[i][j]=2*x*y/max(x+y,1e-30)
    pairs=[sorted([j for j in range(n) if scores[i][j]>0],key=lambda j:(-scores[i][j],j))[:8] for i in range(n)]
    return pairs,dict(pairs=pairs,scores=scores,directional=rows,rule='PROTOCOL.json:pairing.new_pairing_only',GT_read=False)


def test():
    from types import SimpleNamespace
    K=torch.tensor([[40.,0.,32.],[0.,40.,32.],[0.,0.,1.]])
    def cam(tx):
        T=torch.eye(4);T[3,0]=tx
        return SimpleNamespace(HWK=(64,64,K),world_view_transform=T)
    a,b=cam(0),cam(-.2);z=torch.full((64,64),2.);m=torch.ones_like(z)
    supported=directional(z,z,m,m,a,b)
    assert supported['fraction']>.8,supported
    assert directional(z,z*.5,m,m,a,b)['supported']==0,'Occluded layer must be rejected'
    assert directional(z,z,m,m*0,a,b)['supported']==0,'Source foreground mask must apply'
    assert directional(z,z,m*0,m,a,b)['supported']==0,'Reference foreground mask must apply'
    bad=z.clone();bad[:]=float('nan')
    assert directional(z,bad,m,m,a,b)['supported']==0,'Invalid source must be rejected'
    pairs,meta=build_covisibility([a,b],[z,z],[m,m]);assert pairs==[[1],[0]]
    assert abs(meta['scores'][0][1]-meta['scores'][1][0])<1e-12
    print('COVISIBILITY_PROJECTED_PLANE_OCCLUSION_MASK_NAN_TESTS_PASSED',flush=True)

if __name__=='__main__':test()
