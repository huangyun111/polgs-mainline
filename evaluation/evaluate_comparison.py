"""Frozen B4_6 complete-triangle evaluator, with explicit precision/recall/tails."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,gc
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import numpy as np
E=Path(__file__).resolve().parent;ROOT=E.parents[1];B=ROOT/'BASELINES_14VIEW_20261103/evidence';OLD=ROOT/'POLAR_PGSR_REPROJECTION_20261106/evidence'
sys.path.insert(0,str(B));from evaluate_batch import mesh,sample,nearest,canonical,sha,atomic
SEEDS=[20261020,20261021,20261022];OUT=E/'evaluation';OUT.mkdir(exist_ok=True)
def items():
    rows=[]
    for scene in ['StandingRabbit','LyingRabbit']:
        for method in ['GS','PolGS','PolGS_plus']:
            old=json.loads((B/'evaluation'/(method+'_'+scene+'.json')).read_text());p=Path(old['path'])
            assert sha(p)==old['hashes']['mesh'];rows.append((scene,method,p,'official_implementation_reused',old))
        if scene=='StandingRabbit':
            previous=ROOT/'POLAR_GRADIENT_ALLOCATION_20261105/evidence'
            rows += [(scene,'Initial_first',previous/'models/Initial/poisson_mesh_8_pruned.ply','same_checkpoint_initial_reused',None),(scene,'Ours',OLD/'models/PGSR_geo_u280/poisson_mesh_8_pruned.ply','FIRST_frozen_endpoint_reused',None),(scene,'Continue280_first',OLD/'models/Control_joint_u280/poisson_mesh_8_pruned.ply','FIRST_control_reused',None)]
        for name,label in [('Initial','Initial'),('Ours_u280','Ours'),('Continue_u280','Continue280'),('Continue_cost','Continue_cost')]:
            label=label+'_timing_replay' if scene=='StandingRabbit' else label
            rows.append((scene,label,E/scene/'models'/name/'poisson_mesh_8_pruned.ply','new_native_export',None))
    extra=E/'EXTERNAL_COMPLETED_MESHES.json'
    if extra.exists():
        rows += [(x['scene'],x['method'],Path(x['mesh']),x['implementation'],None) for x in json.loads(extra.read_text())]
    return rows
def evaluate(scene,method,path,origin,old):
    gtpath=ROOT/'B4_6/evaluation'/scene/'gt.ply';ap=gtpath.parent/'alignment.json'
    hashes=dict(mesh=sha(path),gt=sha(gtpath),alignment=sha(ap));dest=OUT/(scene+'_'+method+'.json')
    if old:assert hashes==old['hashes']
    if dest.exists():
        result=json.loads(dest.read_text());assert result['hashes']==hashes;return result
    t=time.monotonic();a=json.loads(ap.read_text());assert a['registration_gate_passed'];T=np.array(a['input_world_to_gt']);assert np.allclose(T@np.array(a['gt_to_input_world']),np.eye(4),atol=1e-8)
    pred=mesh(path,T);gt=mesh(gtpath);ps=canonical(pred);gs=canonical(gt);rep=[]
    assert len(pred['f'])>0
    for seed in SEEDS:
        pp,_=sample(ps,100000,seed);gp,_=sample(gs,100000,seed);acc,ng=nearest(gt,pp);comp,npred=nearest(pred,gp);_,pn=nearest(pred,pp);_,gn=nearest(gt,gp)
        ae,ce=acc*1000,comp*1000;assert np.isfinite(ae).all() and np.isfinite(ce).all()
        r=dict(seed=seed,accuracy=float(ae.mean()),completeness=float(ce.mean()),cd=float((ae.mean()+ce.mean())*.5),normal_unsigned_deg=float((np.degrees(np.arccos(np.clip(abs((pn*ng).sum(1)),0,1))).mean()+np.degrees(np.arccos(np.clip(abs((gn*npred).sum(1)),0,1))).mean())*.5))
        for k,x in [('accuracy',ae),('completeness',ce)]:
            for q in [.9,.95,.99]:r[k+'_p'+str(int(q*100))]=float(np.quantile(x,q))
            r[k+'_max']=float(x.max())
            for threshold in [2,5]:
                r[k+'_tail_ge'+str(threshold)+'_fraction']=float((x>=threshold).mean());r[k+'_tail_ge'+str(threshold)+'_mean_contribution']=float(x[x>=threshold].sum()/len(x))
        for threshold in [.5,1.,2.]:
            precision=float((ae<threshold).mean());recall=float((ce<threshold).mean());r['precision_'+str(threshold)]=precision;r['recall_'+str(threshold)]=recall;r['fscore_'+str(threshold)]=2*precision*recall/max(precision+recall,1e-12)
        np.savez_compressed(OUT/(scene+'_'+method+'_'+str(seed)+'_distances.npz'),accuracy_gt_x1000=ae,completeness_gt_x1000=ce)
        rep.append(r)
    result=dict(state='COMPLETED',scene=scene,method=method,implementation=origin,path=str(path),hashes=hashes,training_seed=0,sampling_seeds=SEEDS,samples_per_direction=100000,vertices=len(pred['v']),faces=len(pred['f']),surface_area_GT_coordinates_squared=float(pred['area'].sum()),replicates=rep,mean={k:float(np.mean([r[k] for r in rep])) for k in rep[0] if k!='seed'},sampling_sd={k:float(np.std([r[k] for r in rep],ddof=1)) for k in rep[0] if k!='seed'},evaluation_wall_seconds=time.monotonic()-t,protocol_sha256=sha(E/'protocol.json'),evaluator_sha256=sha(B/'evaluate_batch.py'),b4_6_sha256=sha(ROOT/'B4_6/scripts/evaluate_b4.py'),distance_unit='GT_coordinates_x1000_not_independently_calibrated_mm',registration_refit=False)
    if old:
        result['reused_metric_max_abs_diff']=max(abs(result['mean'][k]-old['mean'][k]) for k in ['cd','accuracy','completeness','normal_unsigned_deg'])
        assert result['reused_metric_max_abs_diff']<1e-9
    assert hashes==dict(mesh=sha(path),gt=sha(gtpath),alignment=sha(ap));atomic(dest,result)
    print(scene,method,result['mean']['cd'],flush=True);del pred,gt,ps,gs;gc.collect();return result
if __name__=='__main__':
    records=[];failures=[]
    for args in items():
        try:records.append(evaluate(*args))
        except Exception:
            err=dict(scene=args[0],method=args[1],state='EVALUATION_FAILED',traceback=traceback.format_exc());atomic(OUT/(args[0]+'_'+args[1]+'_FAILURE.json'),err);failures.append(err);print('EVALUATION_FAILED_CONTINUING',args[0],args[1],flush=True)
    atomic(E/'ALL_COMPLETE_SURFACE_METRICS.json',records)
    atomic(E/'EVALUATION_COMPLETE.json',dict(state='ALL_LISTED_MESHES_EVALUATED' if not failures else 'ALL_LISTED_MESHES_PROCESSED_WITH_EXPLICIT_FAILURES',count=len(records),failures=failures,mesh_hashes={r['scene']+'/'+r['method']:r['hashes']['mesh'] for r in records}))
