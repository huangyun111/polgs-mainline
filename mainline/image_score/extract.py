from pathlib import Path
import hashlib,json,os,runpy,shutil,sys,time,traceback
R=Path('/home/hy/scap_controls_20261111');scene,arm=sys.argv[1:3];W=R/'runs'/scene/arm;C=Path('/home/hy/official_view_ns_20261109/code')
trained=W/'model';out=R/'exports'/scene/arm;out.mkdir(parents=True,exist_ok=False)
(out/'point_cloud').symlink_to(trained/'point_cloud',target_is_directory=True);shutil.copy2(trained/'cfg_args',out/'cfg_args')
os.chdir(C);sys.path.insert(0,str(C));import pymeshlab,torch
torch.cuda.reset_peak_memory_stats();original=pymeshlab.MeshSet.generate_surface_reconstruction_screened_poisson;calls=[]
def bounded(self,*args,**kwargs):
    old=kwargs.get('threads');kwargs['threads']=4;calls.append(dict(kwargs,original_threads=old));return original(self,*args,**kwargs)
pymeshlab.MeshSet.generate_surface_reconstruction_screened_poisson=bounded
sys.argv=[str(C/'render.py'),'-m',str(out),'--resolution','2' if scene=='StandingRabbit' else '1','--iteration','15000','--img','--skip_test','--depth','8']
start=time.monotonic();rec=dict(scene=scene,arm=arm,iteration=15000,argv=sys.argv,source_model=str(trained),native_PLY_before_operations=True,source_sha=hashlib.sha256((C/'render.py').read_bytes()).hexdigest())
try:
    runpy.run_path(str(C/'render.py'),run_name='__main__');mesh=out/'poisson_mesh_8_pruned.ply';assert mesh.stat().st_size>1000
    rec.update(status='COMPLETED',mesh=str(mesh),mesh_bytes=mesh.stat().st_size)
except BaseException:rec.update(status='FAILED',traceback=traceback.format_exc());raise
finally:
    rec.update(wall_seconds=time.monotonic()-start,poisson_calls=calls,peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated());(out/'EXTRACTION_RECEIPT.json').write_text(json.dumps(rec,indent=2))
