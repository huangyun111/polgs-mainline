"""Independent CPU full-surface metrics; no registration to predictions."""
import argparse,hashlib,json,time,os
from pathlib import Path
import numpy as np
import open3d as o3d
ROOT=Path(os.environ.get('B4_ROOT','/home/hy/gaussian_surfels_stage4_20261020'))
BUNDLE=Path('/home/hy/polgs_repro/PolGS_Offline_Bundle')

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def mesh(path,transform=None):
    m=o3d.io.read_triangle_mesh(str(path))
    if transform is not None:m.transform(transform)
    m.compute_vertex_normals();v=np.asarray(m.vertices).copy();f=np.asarray(m.triangles).astype(np.int64);n=np.asarray(m.vertex_normals).copy()
    assert len(v)>0 and len(f)>0 and np.isfinite(v).all()
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
    scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    return {'mesh':m,'v':v,'f':f,'n':n,'area':area,'scene':scene}
def sample(m,count,seed):
    rng=np.random.RandomState(seed);fi=rng.choice(len(m['f']),count,p=m['area']/m['area'].sum());u=rng.rand(count,2);a=np.sqrt(u[:,0])
    bary=np.stack((1-a,a*(1-u[:,1]),a*u[:,1]),1);tri=m['f'][fi]
    xyz=(m['v'][tri]*bary[:,:,None]).sum(1);n=(m['n'][tri]*bary[:,:,None]).sum(1);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    return xyz,n
def nearest(m,p):
    out=m['scene'].compute_closest_points(o3d.core.Tensor(p.astype(np.float32)),nthreads=4)
    q=out['points'].numpy();fi=out['primitive_ids'].numpy().astype(np.int64);uv=out['primitive_uvs'].numpy();bary=np.stack((1-uv[:,0]-uv[:,1],uv[:,0],uv[:,1]),1)
    n=(m['n'][m['f'][fi]]*bary[:,:,None]).sum(1);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    return np.linalg.norm(p-q,axis=1),n
def main(arm):
    started=time.perf_counter();run=ROOT/'runs'/arm
    assert json.loads((run/'train.json').read_text())['state']=='COMPLETED'
    p=ROOT/'evaluation_alignment.json'
    if not p.exists():
        reference=Path('/home/hy/polgs_repro/geometry_eval_20261015/alignment.json')
        p.write_bytes(reference.read_bytes())
    alignment=json.loads(p.read_text());T=np.asarray(alignment['input_world_to_gt']);Ti=np.asarray(alignment['gt_to_input_world'])
    assert np.allclose(T@Ti,np.eye(4),atol=1e-8)
    pred_path=run/'model/poisson_mesh_10_pruned.ply';gt_path=BUNDLE/'data/GT/PISR/StandingRabbit/standing_rabbit.ply'
    pred=mesh(pred_path,T);gt=mesh(gt_path)
    rows=[]
    for seed in [20261020,20261021,20261022]:
        pp,pn=sample(pred,100000,seed);gp,gn=sample(gt,100000,seed)
        acc,ng=nearest(gt,pp);comp,npred=nearest(pred,gp)
        pmae=np.degrees(np.arccos(np.clip(np.abs((pn*ng).sum(1)),0,1))).mean()
        gmae=np.degrees(np.arccos(np.clip(np.abs((gn*npred).sum(1)),0,1))).mean()
        row={'seed':seed,'accuracy_gt_units':float(acc.mean()),'completeness_gt_units':float(comp.mean()),'cd_gt_units':float((acc.mean()+comp.mean())/2),
            'accuracy_mm_author_convention':float(acc.mean()*1000),'completeness_mm_author_convention':float(comp.mean()*1000),'cd_mm_author_convention':float((acc.mean()+comp.mean())*500),
            'mesh_normal_mae_unsigned_deg':float((pmae+gmae)/2)}
        for mm in [.5,1,2]:
            precision=float((acc*1000<mm).mean());recall=float((comp*1000<mm).mean());row['fscore_'+str(mm)+'mm']=2*precision*recall/max(precision+recall,1e-12)
        rows.append(row)
    scores=[json.loads(line) for line in (run/'model/psnr_per_view.jsonl').read_text().splitlines()]
    splits=json.loads((ROOT/'data/StandingRabbit/splits.json').read_text());psnr={}
    for split,ids in splits.items():
        values={d['view']:d['psnr'] for d in scores if int(d['view']) in ids}
        assert len(values)==len(ids),(split,len(values),len(ids))
        psnr[split]={'mean_db':float(np.mean(list(values.values()))),'per_view_db':values,'protocol':'B1 render.py black background/masked GT/full image float64'}
    counts=np.asarray(pred['mesh'].cluster_connected_triangles()[1]);result={'arm':arm,'state':'COMPLETED','samples_per_surface_per_seed':100000,'replicates':rows,
        'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='seed'},'sampling_std':{k:float(np.std([r[k] for r in rows])) for k in rows[0] if k!='seed'},
        'mesh':{'vertices':len(pred['v']),'faces':len(pred['f']),'components':len(counts),'largest_component_face_fraction':float(counts.max()/counts.sum()),'surface_area_gt_units_squared':float(pred['area'].sum()),'watertight':bool(pred['mesh'].is_watertight())},
        'psnr':psnr,'wall_seconds':time.perf_counter()-started,'mesh_sha256':sha(pred_path),'gt_sha256':sha(gt_path),'alignment_sha256':sha(p),
        'registration':'fixed camera/mask-only common similarity from20261015; no predicted-mesh fitting; evaluation-only',
        'official_protocol_equivalent':False,'units':'raw GT coordinates;1000x reported as author-convention mm, not independently physically calibrated',
        'normal_metric':'unsigned symmetric nearest-surface mesh normal MAE, not per-image GT normal MAE',
        'reflective_region_metric':None,'textureless_region_metric':None,'region_reason':'no independently verified semantic region mask; no GT/error-based selection',
        'distance':'unsquared L2 closest triangle distance; CD arithmetic mean of two area-uniform directed means; whole surfaces, no GT trimming'}
    atomic(run/'metrics.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args();main(a.arm)
