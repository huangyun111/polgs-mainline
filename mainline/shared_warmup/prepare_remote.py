from pathlib import Path
import argparse,hashlib,json,shutil,sys,subprocess,os
import numpy as np
R=Path('/home/hy/official_view_ns_20261109');E=R/'setup';OFF=Path('/home/hy/polgs_plus_official_save_audit_20261108')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not (E/'PREPARED.json').exists()
tree=json.loads((OFF/'setup/official_tree.json').read_text())
shutil.copytree(OFF/'code',R/'code',ignore=shutil.ignore_patterns('__pycache__','.git','build','*.egg-info'))
for row in tree:
    p=R/'code'/row['path']
    if p.exists():
        b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['sha'],row['path']
sys.path.insert(0,str(R/'code'))
from arguments import ModelParams,PipelineParams,OptimizationParams
record=dict(scenes={},official_commit='879cc86c86fce873d9a8a9c1e4e200b5961d04d0',native_sources={str(p.relative_to(R/'code')):sha(p) for p in (R/'code').rglob('*') if p.is_file() and p.suffix in ['.py','.cu','.h','.cpp','.cuh']})
for scene,n,source,res,az in [('StandingRabbit',20,Path('/home/hy/polgs_repro/PolGS_Offline_Bundle/data/PolGS_plus/PISR/StandingRabbit'),2,.05),('smvp3d_david',36,OFF/'data/david',1,.5)]:
    data=R/'data'/scene;data.mkdir(parents=True)
    meta=json.loads((E/'RABBIT_OFFICIAL_MANIFEST.json').read_text()) if n==20 else json.loads((OFF/'setup/OFFICIAL_DATA_MANIFEST.json').read_text())
    files=meta['files'];used=[]
    for x in files:
        rel=x['path']
        if Path(rel).parts[0] not in ['image','input_azimuth_maps','s0','s1','s2','mask','cameras.npz']:continue
        src=source/rel;assert sha(src)==x['sha256'],str(src)
        dst=data/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src)
        used.append(dict(path=rel,sha256=x['sha256'],bytes=src.stat().st_size,source=str(src)))
    imgs=sorted((data/'image').glob('*.png'));assert len(imgs)==n
    cams=np.load(data/'cameras.npz');assert {k for k in cams.files if k.startswith('world_mat_')}=={f'world_mat_{i}' for i in range(n)}
    if n==20:mapping=meta['source_mapping']
    else:
        old={x['path']:x['source'] for x in json.loads((OFF/'setup/STAGING.json').read_text())['mapping']}
        mapping=[dict(official_id=i,source_id=int(Path(old[f'image/{i:04d}.png']).stem),original_image=old[f'image/{i:04d}.png']) for i in range(n)]
    assert [x['official_id'] for x in mapping]==list(range(n))
    parser=argparse.ArgumentParser();mp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
    args=parser.parse_args(['-s',str(data),'-m','FROZEN_MODEL_PATH','--resolution',str(res),'--iterations','15000','--stokes_l','0.1','--azimuth_l',str(az),'--azimuth_start','1000'])
    model=vars(mp.extract(args));model.pop('model_path');opt=vars(op.extract(args));pipe=vars(pp.extract(args))
    # Official native defaults are frozen as actually parsed, including position_lr_max_steps=60000.
    record['scenes'][scene]=dict(data=str(data),code=str(R/'code'),views=n,resolution=res,azimuth_l=az,source_mapping=mapping,input_manifest=used,model_params=model,optimization=opt,pipeline=pipe)
    for arm in ['warmup','probe_N','probe_S','N','S']:
        w=R/'runs'/scene/arm;(w/'support').mkdir(parents=True)
        for name in ['geo_renderer.py','slope_proxy.py','extension_loader.py','multiview_geometry.py','covisibility.py','density_state.py','routing_core.py']:shutil.copy2(E/name,w/'support'/name)
        put(w/'DATA.json',dict(scene=scene,arm=arm,data=str(data),code=str(R/'code'),views=n,resolution=res,azimuth_l=az))
    (R/'shared'/scene).mkdir(parents=True)
put(E/'PREPARED.json',record)
shutil.copy2(OFF/'setup/FRAME_CHECK.json',E/'DAVID_FRAME_CHECK.json')
protocol=dict(experiment=R.name,scenes={k:{x:v[x] for x in ['views','resolution','azimuth_l','optimization','model_params','pipeline','source_mapping']} for k,v in record['scenes'].items()},seed=0,arms={'N':'Adam common; stats common','S':'Adam common; stats joint'},shared_prefix_Adam=1000,new_Adam_per_arm=13999,total_Adam_per_trajectory=14999,geometry_weight=.03,geometry_start=1001,geometry='Unchanged validated corrected camera-z PGSR; frozen covisibility top1 from this scene fresh native1000 depths; GT forbidden',route='Identical dual reference + neighbor renders, common autograd.grad, joint.backward, Adam common for both; only native stats reference XY/global scaling selects common/joint; raw rotation/opacity/visibility/topology/reset/order unchanged',endpoint='Only15000 native PLY/map BEFORE final topology; PRE/POST/END complete states and native checkpoint preserved; no280 refinement',extraction={'StandingRabbit':{'resolution':2,'depth':8,'img':True},'smvp3d_david':{'resolution':1,'depth':8,'img':True},'Poisson_threads':4},evaluation='Existing certified full triangle evaluators;100000 area samples/direction,3 fixed sampling seeds; noGT fit/crop/endpoint selection; units scene-specific,not calibrated mm',resources='dl114 only;GPU0 faulty excluded;prefer idle healthy GPU, user allows roomy low-util shared;one worker per scene,N/S sequential on same physical GPU;8192MiB budget plus max2048MiB/10percent reserve;shared lock/registry',scope='Exactly2scenes x2arms seed0,no sweeps/seeds/scenes',monitor='Exit after stable real training and route/shared-state checks;finite queue train->extract->evaluate->verify;startup is not completion')
put(E/'PROTOCOL.json',protocol);put(E/'protocol.json',protocol);print('PREPARED_OFFICIAL_VIEWS',[(k,v['views'],v['source_mapping']) for k,v in record['scenes'].items()])
