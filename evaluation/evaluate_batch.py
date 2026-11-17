"""Same full, untrimmed triangle-distance evaluator for six baselines and M3."""
import os,sys,json,time,csv
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
ROOT=Path('/home/hy/gaussian_surfels_stage4_20261020');E=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'B4_6/scripts'))
from evaluate_b4 import mesh,sample,nearest,sha,atomic
SEEDS=[20261020,20261021,20261022]
def canonical(m):
 v,inv=np.unique(m['v'],axis=0,return_inverse=True);f=np.sort(inv[m['f']],axis=1);f=f[np.lexsort(f.T[::-1])]
 t=v[f];area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)*.5
 return dict(v=v,f=f,area=area,n=np.zeros_like(v))
def path_for(method,scene):
 if method=='M3':return ROOT/'B4_TRADITIONAL/runs'/f'{scene}_seed0_M3/model/poisson_mesh_10_pruned.ply'
 run=E/'runs'/f'{method}_{scene}_seed0';s=json.loads((run/'status.json').read_text());assert s['state']=='COMPLETED',s
 return Path(s['mesh'])
def evaluate(method,scene):
 out=E/'evaluation';out.mkdir(exist_ok=True);dest=out/f'{method}_{scene}.json'
 path=path_for(method,scene);ep=ROOT/'B4_6/evaluation'/scene;gtpath=ep/'gt.ply';ap=ep/'alignment.json'
 hashes={'mesh':sha(path),'gt':sha(gtpath),'alignment':sha(ap)}
 if dest.exists():
  old=json.loads(dest.read_text());assert old['hashes']==hashes;return old
 start=time.monotonic();a=json.loads(ap.read_text());assert a['registration_gate_passed']
 T=np.asarray(a['input_world_to_gt']);assert np.allclose(T@np.asarray(a['gt_to_input_world']),np.eye(4),atol=1e-8)
 pred=mesh(path,T);gt=mesh(gtpath);ps=canonical(pred);gs=canonical(gt);rows=[]
 for seed in SEEDS:
  pp,_=sample(ps,100000,seed);gp,_=sample(gs,100000,seed)
  acc,ng=nearest(gt,pp);comp,npred=nearest(pred,gp)
  _,pn=nearest(pred,pp);_,gn=nearest(gt,gp)
  pmae=np.degrees(np.arccos(np.clip(np.abs((pn*ng).sum(1)),0,1))).mean();gmae=np.degrees(np.arccos(np.clip(np.abs((gn*npred).sum(1)),0,1))).mean()
  row={'sampling_seed':seed,'accuracy':float(acc.mean()*1000),'completeness':float(comp.mean()*1000),'cd':float((acc.mean()+comp.mean())*500),'normal_unsigned_deg':float((pmae+gmae)/2)}
  for threshold in [.5,1.,2.]:
   p=float((acc*1000<threshold).mean());r=float((comp*1000<threshold).mean());row[f'fscore_{threshold}']=2*p*r/max(p+r,1e-12)
  rows.append(row)
  np.savez_compressed(out/f'{method}_{scene}_{seed}_distances.npz',accuracy=acc,completeness=comp)
 result={'state':'COMPLETED','method':method,'scene':scene,'training_seed':0,'path':str(path),'hashes':hashes,'samples_per_direction':100000,'sampling_seeds':SEEDS,'replicates':rows,'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='sampling_seed'},'sampling_sd':{k:float(np.std([r[k] for r in rows],ddof=1)) for k in rows[0] if k!='sampling_seed'},'vertices':len(pred['v']),'faces':len(pred['f']),'wall_seconds':time.monotonic()-start,'protocol':'Full untrimmed surfaces; canonical vertex/corner/face ordering on BOTH meshes; area samples to closest triangle; symmetric unsigned interpolated surface normals; GT coordinates x1000, not calibrated physical mm; no prediction registration; no GT used for training or selection'}
 assert hashes=={'mesh':sha(path),'gt':sha(gtpath),'alignment':sha(ap)}
 atomic(dest,result);print('EVALUATED',method,scene,result['mean'],flush=True);return result
if __name__=='__main__':
 methods=sys.argv[1:] or ['GS','PolGS','PolGS_plus','M3']
 for scene in ['StandingRabbit','LyingRabbit']:
  for method in methods:evaluate(method,scene)
