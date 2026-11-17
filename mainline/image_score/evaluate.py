from pathlib import Path
import hashlib,importlib.util,json,sys,time
R=Path('/home/hy/scap_controls_20261111');scene,arm=sys.argv[1:3];W=R/'runs'/scene/arm;out=R/'evaluation'/scene;out.mkdir(parents=True,exist_ok=True)
seal=R/'setup/REFERENCE_REUSE.json';assert json.loads(seal.read_text())['passed']
pred=R/'exports'/scene/arm/'poisson_mesh_8_pruned.ply';start=time.monotonic()
spec=importlib.util.spec_from_file_location('rabbit','/home/hy/gaussian_surfels_stage4_20261020/TWO_RABBIT_COMPARISON_20261107/evidence/evaluate_comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=out;m.E=Path('/home/hy/official_view_ns_20261109/setup')
# Reuse registered immutable GT/alignment identity; hash each new prediction once.
oldreceipt=Path('/home/hy/polar_structure_content_20261110/runs/D2/EVALUATION_RECEIPT.json')
registered=json.loads(oldreceipt.read_text());native_sha=m.sha;identity_cache={}
gt=m.ROOT/'B4_6/evaluation/StandingRabbit/gt.ply';align=gt.parent/'alignment.json'
for path,key in [(gt,'gt'),(align,'alignment')]:
    if path.stat().st_mtime_ns<=oldreceipt.stat().st_mtime_ns:identity_cache[(str(path),path.stat().st_size,path.stat().st_mtime_ns)]=registered['hashes'][key]
def cached_sha(path):
    path=Path(path);k=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
    if k not in identity_cache:identity_cache[k]=native_sha(path)
    return identity_cache[k]
m.sha=cached_sha
row=m.evaluate('StandingRabbit',arm,pred,'S_cap_fixed_rules_stage2_native15000',None)
row.update(evaluation_wall_seconds=time.monotonic()-start,training_seed=0,view_protocol='official20',area_repeats_not_training_seeds=True,structure_seal_sha=hashlib.sha256(seal.read_bytes()).hexdigest(),diagnostic_protocol_sha=hashlib.sha256((R/'setup/PROTOCOL.md').read_bytes()).hexdigest())
(W/'EVALUATION_RECEIPT.json').write_text(json.dumps(row,indent=2));print('EVALUATED',arm,row['mean'],flush=True)
