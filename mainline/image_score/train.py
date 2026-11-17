"""Instrument and extend the original PolGS++ train.py at explicit hook points."""
from pathlib import Path
import contextlib,hashlib,json,os,random,sys,time,traceback,faulthandler
import numpy as np
import torch
R=Path('/home/hy/scap_controls_20261111');OLD=Path('/home/hy/official_view_ns_20261109');scene,arm=sys.argv[1:3]
W=R/'runs'/scene/arm;info=json.loads((W/'DATA.json').read_text());CODE=Path(info['code'])
sys.path[:0]=[str(R/'setup'),str(CODE)];os.chdir(CODE)
from density_state import model_state,rng_state,restore_rng,fingerprint
def put(name,value):
    p=W/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(t):return dict(shape=list(t.shape),sha256=hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0])
        if Path(p).name in {'gt.ply','alignment.json','GT_METADATA.json','GT_ARRAYS.npz'} or '/gt_normal/' in p or 'gt_mesh_original' in p:
            raise RuntimeError('Forbidden GT read during training: '+p)
sys.addaudithook(audit)
faulthandler.enable();faulthandler.dump_traceback_later(180,repeat=True)
START=time.monotonic();state=dict(iteration=0,adam_calls=0,geometry_calls=0,nonempty_calls=0,valid_pixels=0,kept_pixels=0,gpu_ms=0.,preprocessing_seconds=0.)
common=hashlib.sha256();viewrng=random.Random(0);pairrng=random.Random(20261106)
bggen=torch.Generator(device='cuda').manual_seed(0);idxgen=torch.Generator(device='cuda').manual_seed(0)
random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
extra_rng=rng_state();extra_rng['python']=random.Random(20261106).getstate()
extra_rng['numpy']=np.random.RandomState(20261106).get_state()
extra_rng['torch']=torch.Generator(device='cpu').manual_seed(20261106).get_state()
extra_rng['cuda']=[torch.Generator(device='cuda').manual_seed(20261106).get_state()]
@contextlib.contextmanager
def extra_stream():
    global extra_rng
    saved=rng_state();restore_rng(extra_rng)
    try:yield
    finally:extra_rng=rng_state();restore_rng(saved)

class ProbeDone(Exception):pass
class GracefulStop(Exception):pass
from density_state import restore_model
from routing_core import gradients,assign_optimizer,statistics_fixture,STATS
SHARED=OLD/'shared'/scene
IS_WARM=False;IS_PROBE=False;LABEL='N' if arm=='N-cap' else 'S'
STATISTICS=True;UPDATE_JOINT=arm=='Joint-update';new_calls=0;ROUTE={};EXPECTED=None;SCORE_INPUT=None
def opt_contract(actual,expected):
    assert actual['stokes_l']==.1 and actual['azimuth_l']==.05
    return {k:v for k,v in actual.items() if k not in ['stokes_l','azimuth_l']}=={k:v for k,v in expected.items() if k not in ['stokes_l','azimuth_l']}

def parameters(g):return [p for group in g.optimizer.param_groups for p in group['params']]
def names(g):return [group['name']+':'+str(i) for group in g.optimizer.param_groups for i,p in enumerate(group['params'])]
def snapshot(s,name):
    g=s['gaussians'];saved=dict(state=model_state(g),rng=rng_state(),iteration=s['iteration'],adam_calls=state['adam_calls'],new_Adam_calls=new_calls,depth_list=s['depth_list'].detach().clone(),bg_rng=bggen.get_state(),idx_rng=idxgen.get_state(),view_rng=viewrng.getstate(),pair_rng=pairrng.getstate(),extra_rng=extra_rng,viewpoint_stack=[views.index(c) for c in s['viewpoint_stack']],common_sha=common.hexdigest(),pairs=pairs,dataset=vars(s['dataset']),opt=vars(s['opt']),pipe=vars(s['pipe']),extent=float(s['scene'].cameras_extent),counts=dict(state))
    if hasattr(g,'_image_score'):saved.update(image_score=g._image_score.detach().clone(),image_views=sorted(g._image_views))
    p=W/(name+'.pth');assert not p.exists();torch.save(saved,p)
    row=dict(iteration=s['iteration'],points=len(g._xyz),no_large_asset_rehash=True,adam_calls=state['adam_calls'],new_Adam_calls=new_calls,common_sha=common.hexdigest(),bytes=p.stat().st_size);put(name+'.json',row);return saved,row
def init(s):
    global native,dual,views,pairs,refpkg,neighborpkg,meta,common,extra_rng
    import gaussian_renderer as gr
    native=gr.render;views=s['scene'].getTrainCameras();pairs=[]
    expected=json.loads((OLD/'setup/PREPARED.json').read_text())['scenes'][scene]
    assert len(views)==info['views'] and len(s['scene'].getTestCameras())==0
    assert [c.uid for c in views]==list(range(info['views']))
    assert [c.image_name for c in views]==[f'{i:04d}' for i in range(info['views'])]
    cams=[dict(name=c.image_name,uid=c.uid) for c in views]
    assert tuple(s['depth_list'].shape)[0]==info['views']
    assert opt_contract(vars(s['opt']),expected['optimization']) and vars(s['pipe'])==expected['pipeline']
    assert {k:v for k,v in vars(s['dataset']).items() if k!='model_path'}==expected['model_params']
    g=s['gaussians'];refpkg=neighborpkg=meta=None
    if IS_WARM:
        put('INITIALIZATION.json',dict(cameras=cams,dataset=vars(s['dataset']),optimization=vars(s['opt']),pipeline=vars(s['pipe']),model_state_sha=fingerprint(model_state(g)),points=len(g._xyz),GT_read=False,depth_cache_shape=list(s['depth_list'].shape),source_mapping=expected['source_mapping']))
    else:
        bundle=torch.load(SHARED/'WARMUP1000.pth');assert bundle['iteration']==1000 and bundle['adam_calls']==1000
        assert opt_contract(vars(s['opt']),bundle['opt']) and vars(s['pipe'])==bundle['pipe']
        original=json.loads((OLD/'runs'/scene/'warmup/INITIALIZATION.json').read_text());assert cams==[{k:x[k] for k in ['name','uid']} for x in original['cameras']]
        restore_model(g,bundle['state'],s['opt']);s['depth_list'].copy_(bundle['depth_list'])
        assert all(torch.equal(getattr(g,k),v) for k,v in zip(['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'],[bundle['state']['capture'][i] for i in [1,2,3,4,5,6]]))
        s['viewpoint_stack'][:]=[views[i] for i in bundle['viewpoint_stack']]
        prefix=[json.loads(x) for x in (SHARED/'prefix_trace.jsonl').read_text().splitlines()]
        assert len(prefix)==1000 and set(x['reference'] for x in prefix)==set(range(info['views']))
        common=hashlib.sha256()
        for row in prefix:common.update(bytes.fromhex(row['sample_sha']))
        assert common.hexdigest()==bundle['common_sha']
        viewrng.setstate(bundle['view_rng']);bggen.set_state(bundle['bg_rng'].cpu());idxgen.set_state(bundle['idx_rng'].cpu());pairrng.setstate(bundle['pair_rng']);extra_rng=bundle['extra_rng']
        pairs=json.loads((SHARED/'FROZEN_PAIRS.json').read_text())['pairs'];assert len(pairs)==info['views'] and all(pairs)
        from geo_renderer import make_renderer
        dual=make_renderer(gr)
        restore_rng(bundle['rng'])
        state.update(iteration=1000,adam_calls=1000)
        restored=dict(passed=True,source=str(SHARED/'WARMUP1000.pth'),registered_source=json.loads((SHARED/'WARMUP1000.json').read_text()),points=len(g._xyz),remaining_stack=bundle['viewpoint_stack'],common_prefix_sha=common.hexdigest(),optimization=vars(s['opt']),prefix_updates=1000,new_updates=0,large_asset_rehash=False)
        put('RESTORE_VERIFICATION.json',restored)
        assert json.loads((SHARED/'PAIR_GATES.json').read_text())['passed']
    original_step=g.optimizer.step
    def step(*args,**kwargs):
        global new_calls
        if not IS_WARM:
            before={id(p):v for p,v in zip(ROUTE['parameter_objects'],ROUTE['selected_optimizer'])}
            for p in parameters(g):
                if id(p) in before:assert p.grad is before[id(p)]
                else:assert p.grad is None,'Native topology replacements preserve None'
            state['optimizer_joint_calls' if UPDATE_JOINT else 'optimizer_common_calls']+=1
        result=original_step(*args,**kwargs);state['adam_calls']+=1
        if not IS_WARM:new_calls+=1
        return result
    g.optimizer.step=step
    # Preserve counters and state at a completed step if the owner requests graceful recovery.
    import signal
    def request_stop(signum,frame):state['graceful_stop_requested']=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    state.update(common_gradient_passes=0,joint_backward_passes=0,common_gradient_ms=0.,joint_backward_ms=0.,neighbor_geometry_ms=0.,optimizer_common_calls=0,optimizer_joint_calls=0,stats_common_calls=0,stats_joint_calls=0,reference_render_ms=0.)
    if not IS_WARM:
        original_stats=g.add_densification_stats;ROUTE['original_stats']=original_stats
        def routed_stats(ref,visible):
            ps=parameters(g);assert ref is refpkg['viewspace_points']
            assert all(p is q for p,q in zip(ps,ROUTE['parameter_objects'])) and all(p.grad is v for p,v in zip(ps,ROUTE['selected_optimizer']))
            oldref=ref.grad;oldscale=g._scaling.grad;versions=[p._version for p in ps];label='joint' if STATISTICS else 'common'
            try:
                ref.grad=ROUTE[label+'_ref'];g._scaling.grad=ROUTE[label][ROUTE['scale_index']];original_stats(ref,visible)
                if arm=='S-dual':
                    from cap_core import common_add
                    common_add(g,ROUTE['common_ref'],ROUTE['common'][ROUTE['scale_index']],visible)
                    if EXPECTED is not None:assert all(torch.equal(g._common_stats[k],EXPECTED['common'][k]) for k in STATS)
                    state['dual_common_stats_calls']=state.get('dual_common_stats_calls',0)+1
                if EXPECTED is not None:
                    assert all(torch.equal(getattr(g,k),EXPECTED[label][k]) for k in STATS)
                    put('FIRST_REAL_STATISTICS_VERIFIED.json',dict(passed=True,iteration=1001,statistics_source=label,optimizer_source='joint' if UPDATE_JOINT else 'common'))
            finally:ref.grad=oldref;g._scaling.grad=oldscale
            assert versions==[p._version for p in ps] and all(p.grad is v for p,v in zip(ps,ROUTE['selected_optimizer']))
            state['stats_joint_calls' if STATISTICS else 'stats_common_calls']+=1
            if arm=='Image-score':
                from image_score import accumulate
                audit=accumulate(g,native,*SCORE_INPUT,state,deep=EXPECTED is not None)
                if audit is not None:put('IMAGE_SCORE_READONLY_ORACLE.json',audit)
        g.add_densification_stats=routed_stats
    from cap_core import install
    install(g,W,R,arm,state,s['opt'])
    torch.cuda.reset_peak_memory_stats();state['training_start']=time.monotonic()
    return 1 if IS_WARM else 1001
def begin(s):
    state['iteration']=s['iteration'];n=len(s['gaussians']._xyz);state['forward_points']=n;state['Gaussian_forward_steps_new']+=n
    if s['iteration']<15000:state['Gaussian_Adam_exposure_new']+=n
    if s['iteration']>1100:assert n<=39000
def render_ref(*args,**kwargs):
    global refpkg
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
    refpkg=(native if IS_WARM else dual)(*args,**kwargs);b.record();b.synchronize();state['reference_render_ms']+=a.elapsed_time(b)
    return refpkg
def add_geometry(s):
    global GEO,COMMON,neighborpkg,meta,SCORE_INPUT
    i=views.index(s['viewpoint_cam']);h=hashlib.sha256();h.update(np.array([s['iteration'],i],dtype=np.int64).tobytes());h.update(s['background'].detach().cpu().numpy().tobytes())
    if not IS_WARM:h.update(s['idx'].detach().cpu().numpy().tobytes())
    sample=h.hexdigest();common.update(bytes.fromhex(sample));state.update(reference=i,neighbor=None,sample_sha=sample,common_loss=float(s['loss'].detach()),geometry_loss=0.,step_valid=0,step_kept=0)
    COMMON=s['loss'];GEO=None
    if arm=='Image-score':SCORE_INPUT=(s['viewpoint_cam'],s['pipe'],s['image'].detach(),s['gt_image'].detach(),s['visibility_filter'],s['depth_list'])
    if IS_WARM:return COMMON
    j=pairs[i][0];state['neighbor']=j
    from multiview_geometry import geometry_loss
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
    with extra_stream():neighborpkg=dual(views[j],s['gaussians'],s['pipe'],s['background'],s['patch_size'],initial_stage=False)
    GEO,meta=geometry_loss(refpkg['geo_depth'],neighborpkg['geo_depth'],views[i],views[j]);b.record();b.synchronize();state['neighbor_geometry_ms']+=a.elapsed_time(b)
    kept=int(meta['mask'].sum());valid=int(meta['valid'].sum());state['geometry_calls']+=1;state['nonempty_calls']+=int(kept>0);state['valid_pixels']+=valid;state['kept_pixels']+=kept
    state.update(geometry_loss=float(GEO.detach()),step_valid=valid,step_kept=kept)
    return COMMON+GEO if UPDATE_JOINT else COMMON
def backward(s):
    global ROUTE,EXPECTED
    if IS_WARM:COMMON.backward();return
    g=s['gaussians'];ps=parameters(g);scale=next(i for i,p in enumerate(ps) if p is g._scaling);first=s['iteration']==1001
    before=[p._version for p in ps] if first else None
    before_rng=fingerprint(rng_state()) if first else None;before_depth=fingerprint(s['depth_list']) if first else None
    result=gradients(COMMON,COMMON+GEO,ps,refpkg['viewspace_points'],neighborpkg['viewspace_points'],oracle=first)
    state['common_gradient_passes']+=1;state['joint_backward_passes']+=1
    state['common_gradient_ms']+=result['common_gradient_ms'];state['joint_backward_ms']+=result['joint_backward_ms']
    selected=assign_optimizer(ps,result,UPDATE_JOINT);original=ROUTE['original_stats'];ROUTE=dict(result,parameter_objects=ps,selected_optimizer=selected,scale_index=scale,original_stats=original);EXPECTED=None
    if first:
        fixture,EXPECTED=statistics_fixture(g,original,refpkg['viewspace_points'],s['visibility_filter'],result['common_ref'],result['joint_ref'],result['common'][scale],result['joint'][scale])
        assert before==[p._version for p in ps] and before_rng==fingerprint(rng_state()) and before_depth==fingerprint(s['depth_list'])
        put('ROUTING_FIXTURE.json',dict(passed=True,arm=LABEL,iteration=1001,new_Adam_calls=new_calls,update_geometry=UPDATE_JOINT,statistics_geometry=STATISTICS,common_gradient_oracle=result['audit'],statistics_oracle=fixture,parameters_Adam_statistics_unchanged=True,global_RNG_and_depth_cache_unchanged=True,reference=state['reference'],neighbor=state['neighbor'],sample_sha=state['sample_sha'],parameter_versions_before=before,parameter_names=names(g)))
        if IS_PROBE:raise ProbeDone()
def operation_before(s):
    state['operation_points_before']=len(s['gaussians']._xyz)
    if s['iteration']==15000:snapshot(s,'PRE_OPERATION_15000')
def operation_after(s):
    with (W/'population_operations.jsonl').open('a') as f:f.write(json.dumps(dict(iteration=s['iteration'],before=state['operation_points_before'],after=len(s['gaussians']._xyz)))+'\n')
    if s['iteration']==15000:snapshot(s,'POST_OPERATION_15000')
def end(s):
    global refpkg,neighborpkg,meta,ROUTE,EXPECTED
    assert bool(torch.isfinite(s['loss']).all());state['gpu_ms']+=float(s['iter_start'].elapsed_time(s['iter_end']))
    row={k:v for k,v in state.items() if k!='training_start'};row.update(points=len(s['gaussians']._xyz),total_loss=float(s['loss'].detach()),common_sha=common.hexdigest(),wall_seconds=time.monotonic()-START,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),new_Adam_calls=new_calls)
    with (W/'trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if s['iteration']%100==0 or s['iteration'] in [1,1001,15000]:put('progress.json',row);print('OFFICIAL_NS_STEP',json.dumps(row),flush=True)
    if IS_WARM and s['iteration']==1000:
        saved,receipt=snapshot(s,'WARMUP1000');import shutil
        shutil.copy2(W/'WARMUP1000.pth',SHARED/'WARMUP1000.pth');shutil.copy2(W/'WARMUP1000.json',SHARED/'WARMUP1000.json');shutil.copy2(W/'trace.jsonl',SHARED/'prefix_trace.jsonl')
        from covisibility import build_covisibility
        start=time.monotonic()
        with extra_stream(),torch.no_grad():
            depths=[native(c,s['gaussians'],s['pipe'],torch.zeros(3,device='cuda'),[float('inf'),float('inf')],initial_stage=False)['depth'].detach().clone() for c in views]
            masks=[c.get_gtMask(s['dataset'].use_mask) for c in views];pp,pm=build_covisibility(views,depths,masks)
            assert len(pp)==info['views'] and all(pp),'Every official view needs a valid frozen top1'
            torch.save(dict(depths=[d.cpu() for d in depths],warmup=receipt),SHARED/'COVISIBILITY_SOURCE_DEPTHS.pth')
        (SHARED/'FROZEN_PAIRS.json').write_text(json.dumps(dict(pairs=pp,GT_read=False,source='new1000freshdepths',top1=True)))
        (SHARED/'FROZEN_COVISIBILITY.json').write_text(json.dumps(dict(**pm,source_model_sha=receipt['state_sha'],source_depths_sha=sha(SHARED/'COVISIBILITY_SOURCE_DEPTHS.pth'),seconds=time.monotonic()-start)))
    if not IS_WARM and s['iteration']%1000==0:snapshot(s,'END_ITER_'+str(s['iteration']))
    if state.get('graceful_stop_requested') and s['iteration']<15000:
        snapshot(s,'GRACEFUL_END_'+str(s['iteration']));raise GracefulStop()
    refpkg=neighborpkg=meta=None;EXPECTED=None
    if not IS_WARM:ROUTE={'original_stats':ROUTE['original_stats']}

source=(CODE/'train.py').read_text();code=source
def replace(old,new):
    global code
    assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
replace('set_seed(953)','set_seed(0)')
replace('    for iteration in range(first_iter, opt.iterations + 2):','    first_iter = _full_init(locals())\n    for iteration in range(first_iter, _stop_iteration + 1):')
replace('        loss = 0\n','        _full_begin(locals())\n        loss = 0\n')
replace('viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))','viewpoint_stack.pop(_view_rng.randint(0, len(viewpoint_stack) - 1))')
replace('torch.rand((3), dtype=torch.float32, device="cuda")','torch.rand((3), dtype=torch.float32, device="cuda", generator=_bg_generator)')
replace('torch.randint(bs.shape[0], [2048])','torch.randint(bs.shape[0], [2048], generator=_idx_generator)')
replace('render_pkg = render(viewpoint_cam','render_pkg = _full_render(viewpoint_cam')
replace('        loss.backward()','        loss = _full_geometry(locals())\n        _route_backward(locals())')
replace('                    gaussians.adaptive_prune(min_opac, scene.cameras_extent)','                    _full_op_before(locals())\n                    gaussians.adaptive_prune(min_opac, scene.cameras_extent)')
replace('                    gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent)','                    gaussians.adaptive_densify(opt.densify_grad_threshold, scene.cameras_extent)\n                    _full_op_after(locals())')
replace('\n\n\ndef save_current_files(args):','\n        _full_end(locals())\n\n\ndef save_current_files(args):')
(W/'executed_train.py').write_text(code)
sys.argv=[str(CODE/'train.py'),'-s',info['data'],'-m',str(W/'model'),'--resolution',str(info['resolution']),'--iterations','15000','--save_iterations','15000','--checkpoint_iterations','15000','--test_iterations','15000','--stokes_l','0.1','--azimuth_l','0.05','--azimuth_start','1000']
receipt=dict(scene=scene,arm=arm,seed=0,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],argv=sys.argv,native_source=str(CODE/'train.py'),executed_source=str(W/'executed_train.py'),explicit_P_enabled=True,GT_read=False)
rc=0
try:
    scope=dict(__name__='__main__',__file__=str(CODE/'train.py'),_full_init=init,_full_begin=begin,_full_render=render_ref,_full_geometry=add_geometry,_route_backward=backward,_full_op_before=operation_before,_full_op_after=operation_after,_full_end=end,_view_rng=viewrng,_bg_generator=bggen,_idx_generator=idxgen,_stop_iteration=1000 if IS_WARM else 15000)
    exec(compile(code,str(W/'executed_train.py'),'exec'),scope)
    assert (state['iteration'],state['adam_calls'])==((1000,1000) if IS_WARM else (15000,14999)),state
    if not IS_WARM:assert (W/'PRE_OPERATION_15000.pth').exists() and (W/'POST_OPERATION_15000.pth').exists()
    receipt['status']='COMPLETED'
except ProbeDone:
    receipt['status']='ZERO_UPDATE_PROBE_PASSED'
except GracefulStop:
    receipt['status']='GRACEFULLY_CHECKPOINTED'
    rc=75
except BaseException:
    rc=1;receipt.update(status='FAILED',traceback=traceback.format_exc());traceback.print_exc()
finally:
    faulthandler.cancel_dump_traceback_later()
    receipt.update(state,new_Adam_calls=state['adam_calls'] if IS_WARM else new_calls,prefix_updates_reused=0 if IS_WARM else 1000,continuation_wall_seconds=time.monotonic()-START,common_sample_sha=common.hexdigest(),wall_seconds=time.monotonic()-START,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),exit_code=rc)
    put('TRAINING_RECEIPT.json',receipt);print('FULLTRAIN_END',json.dumps(receipt),flush=True)
sys.exit(rc)
