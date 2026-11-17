from pathlib import Path
import datetime,json,os,subprocess,sys,time,traceback
R=Path('/home/hy/official_view_ns_20261109');E=R/'setup';scene,gpu,uuid=sys.argv[1:4];W=R/'runs'/scene
PY='/home/hy/gaussian_surfels_repro/envs/gaussian_surfels/bin/python';CUDA='/home/hy/gaussian_surfels_repro/toolchains/cuda11.8'
env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=uuid,CUDA_HOME=CUDA,PATH=str(Path(PY).parent)+':'+CUDA+'/bin:'+env['PATH'],CPATH=CUDA+'/include:/usr/local/cuda-11.3/targets/x86_64-linux/include',LD_LIBRARY_PATH=str(Path(PY).parents[1]/'lib')+':'+CUDA+'/lib64',PYTHONPATH='/home/hy/polgs_repro/PolGS_Offline_Bundle/runtime',TORCH_EXTENSIONS_DIR='/home/hy/gaussian_surfels_stage4_20261020/BASELINES_14VIEW_20261103/evidence/jit_cache/PolGS_plus',TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def put(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
start=time.monotonic();rec=dict(scene=scene,gpu=int(gpu),gpu_uuid=uuid,pid=os.getpid(),status='RUNNING',python=PY,start_utc=now());put(W/'WORKER.json',rec)
def run(script,args,log,**kwargs):
    rec.update(stage=script,args=args,updated_utc=now());put(W/'WORKER.json',rec)
    with log.open('xb') as f:
        p=subprocess.Popen([PY,'-u',str(E/script),*args],env=kwargs.get('env',env),stdout=f,stderr=subprocess.STDOUT)
        put(W/'ACTIVE_CHILD.json',dict(stage=script,args=args,pid=p.pid,log=str(log),time=now()))
        rc=p.wait()
    assert rc==0,(script,args,rc,str(log))
try:
    run('train.py',[scene,'warmup'],W/'warmup/train.log')
    for arm in ['N','S']:run('train.py',[scene,'probe_'+arm],W/('probe_'+arm)/'train.log')
    initials={a:read(W/('probe_'+a)/'RESTORE_VERIFICATION.json') for a in ['N','S']};assert initials['N']==initials['S']
    fixtures={a:read(W/('probe_'+a)/'ROUTING_FIXTURE.json') for a in ['N','S']}
    for a,f in fixtures.items():assert f['passed'] and f['new_Adam_calls']==0 and not f['update_geometry'] and f['statistics_geometry']==(a=='S')
    for key in ['reference','neighbor','sample_sha','model_state_without_gradients_sha','parameter_names']:assert fixtures['N'][key]==fixtures['S'][key]
    put(R/'shared'/scene/'PAIR_GATES.json',dict(passed=True,time=now(),restore=initials['N'],fixtures=fixtures,same_physical_gpu=uuid,probe_updates=0))
    rec.update(status='WARMUP_AND_PAIR_GATES_READY',updated_utc=now());put(W/'WARMUP_READY.json',rec)
except BaseException:
    rec.update(status='FAILED',traceback=traceback.format_exc(),updated_utc=now());put(W/'WARMUP_FAILED.json',rec);raise
