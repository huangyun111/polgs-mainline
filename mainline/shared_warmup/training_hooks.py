from density_state import restore_model
from routing_core import gradients,assign_optimizer,statistics_fixture,STATS
SHARED=R/'shared'/scene
IS_WARM=arm=='warmup';IS_PROBE=arm.startswith('probe_');LABEL=arm[-1] if IS_PROBE else arm
STATISTICS=LABEL=='S';new_calls=0;ROUTE={};EXPECTED=None
def parameters(g):return [p for group in g.optimizer.param_groups for p in group['params']]
def names(g):return [group['name']+':'+str(i) for group in g.optimizer.param_groups for i,p in enumerate(group['params'])]
def snapshot(s,name):
    g=s['gaussians'];saved=dict(state=model_state(g),rng=rng_state(),iteration=s['iteration'],adam_calls=state['adam_calls'],new_Adam_calls=new_calls,depth_list=s['depth_list'].detach().clone(),bg_rng=bggen.get_state(),idx_rng=idxgen.get_state(),view_rng=viewrng.getstate(),pair_rng=pairrng.getstate(),extra_rng=extra_rng,viewpoint_stack=[views.index(c) for c in s['viewpoint_stack']],common_sha=common.hexdigest(),pairs=pairs,dataset=vars(s['dataset']),opt=vars(s['opt']),pipe=vars(s['pipe']),extent=float(s['scene'].cameras_extent),counts=dict(state))
    p=W/(name+'.pth');assert not p.exists();torch.save(saved,p)
    row=dict(iteration=s['iteration'],points=len(g._xyz),state_sha=fingerprint(saved['state']),rng_sha=fingerprint(saved['rng']),adam_calls=state['adam_calls'],new_Adam_calls=new_calls,common_sha=common.hexdigest(),bytes=p.stat().st_size,sha256=sha(p));put(name+'.json',row);return saved,row
def init(s):
    global native,dual,views,pairs,refpkg,neighborpkg,meta,common,extra_rng
    import gaussian_renderer as gr
    native=gr.render;views=s['scene'].getTrainCameras();pairs=[]
    expected=json.loads((R/'setup/PREPARED.json').read_text())['scenes'][scene]
    assert len(views)==info['views'] and len(s['scene'].getTestCameras())==0
    assert [c.uid for c in views]==list(range(info['views']))
    assert [c.image_name for c in views]==[f'{i:04d}' for i in range(info['views'])]
    cams=[dict(name=c.image_name,uid=c.uid,**{k:digest(getattr(c,k)) for k in ['world_view_transform','projection_matrix','mask','original_image','s0','s1','s2'] if torch.is_tensor(getattr(c,k,None))}) for c in views]
    assert tuple(s['depth_list'].shape)[0]==info['views']
    assert vars(s['opt'])==expected['optimization'] and vars(s['pipe'])==expected['pipeline']
    assert {k:v for k,v in vars(s['dataset']).items() if k!='model_path'}==expected['model_params']
    g=s['gaussians'];refpkg=neighborpkg=meta=None
    if IS_WARM:
        put('INITIALIZATION.json',dict(cameras=cams,dataset=vars(s['dataset']),optimization=vars(s['opt']),pipeline=vars(s['pipe']),model_state_sha=fingerprint(model_state(g)),points=len(g._xyz),GT_read=False,depth_cache_shape=list(s['depth_list'].shape),source_mapping=expected['source_mapping']))
    else:
        bundle=torch.load(SHARED/'WARMUP1000.pth');assert bundle['iteration']==1000 and bundle['adam_calls']==1000
        assert vars(s['opt'])==bundle['opt'] and vars(s['pipe'])==bundle['pipe']
        original=json.loads((R/'runs'/scene/'warmup/INITIALIZATION.json').read_text());assert cams==original['cameras']
        restore_model(g,bundle['state'],s['opt']);s['depth_list'].copy_(bundle['depth_list'])
        assert fingerprint(model_state(g))==fingerprint(bundle['state'])
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
        restored=dict(source_file_sha=sha(SHARED/'WARMUP1000.pth'),model_state_sha=fingerprint(model_state(g)),depth_cache_sha=fingerprint(s['depth_list']),random_states_sha=fingerprint(dict(global_rng=rng_state(),view_rng=viewrng.getstate(),bg_rng=bggen.get_state(),idx_rng=idxgen.get_state(),pair_rng=pairrng.getstate(),extra_rng=extra_rng)),remaining_stack=bundle['viewpoint_stack'],common_prefix_sha=common.hexdigest(),optimization_sha=fingerprint(vars(s['opt'])),pipeline_sha=fingerprint(vars(s['pipe'])),cameras_sha=fingerprint(cams),points=len(g._xyz),prefix_updates=1000,new_updates=0)
        put('RESTORE_VERIFICATION.json',restored)
        if not IS_PROBE:
            gate=json.loads((SHARED/'PAIR_GATES.json').read_text());assert gate['passed']
            assert restored==gate['restore'],'Formal restoration must equal both zero-update probes'
    original_step=g.optimizer.step
    def step(*args,**kwargs):
        global new_calls
        if not IS_WARM:
            before={id(p):v for p,v in zip(ROUTE['parameter_objects'],ROUTE['selected_optimizer'])}
            for p in parameters(g):
                if id(p) in before:assert p.grad is before[id(p)]
                else:assert p.grad is None,'Native topology replacements preserve None'
            state['optimizer_common_calls']+=1
        result=original_step(*args,**kwargs);state['adam_calls']+=1
        if not IS_WARM:new_calls+=1
        return result
    g.optimizer.step=step
    state.update(common_gradient_passes=0,joint_backward_passes=0,common_gradient_ms=0.,joint_backward_ms=0.,neighbor_geometry_ms=0.,optimizer_common_calls=0,stats_common_calls=0,stats_joint_calls=0,reference_render_ms=0.)
    if not IS_WARM:
        original_stats=g.add_densification_stats;ROUTE['original_stats']=original_stats
        def routed_stats(ref,visible):
            ps=parameters(g);assert ref is refpkg['viewspace_points']
            assert all(p is q for p,q in zip(ps,ROUTE['parameter_objects'])) and all(p.grad is v for p,v in zip(ps,ROUTE['selected_optimizer']))
            oldref=ref.grad;oldscale=g._scaling.grad;versions=[p._version for p in ps];label='joint' if STATISTICS else 'common'
            try:
                ref.grad=ROUTE[label+'_ref'];g._scaling.grad=ROUTE[label][ROUTE['scale_index']];original_stats(ref,visible)
                if EXPECTED is not None:
                    assert all(torch.equal(getattr(g,k),EXPECTED[label][k]) for k in STATS)
                    put('FIRST_REAL_STATISTICS_VERIFIED.json',dict(passed=True,iteration=1001,statistics_source=label,optimizer_source='common'))
            finally:ref.grad=oldref;g._scaling.grad=oldscale
            assert versions==[p._version for p in ps] and all(p.grad is v for p,v in zip(ps,ROUTE['selected_optimizer']))
            state['stats_joint_calls' if STATISTICS else 'stats_common_calls']+=1
        g.add_densification_stats=routed_stats
    torch.cuda.reset_peak_memory_stats();state['training_start']=time.monotonic()
    return 1 if IS_WARM else 1001
def begin(s):state['iteration']=s['iteration']
def render_ref(*args,**kwargs):
    global refpkg
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
    refpkg=(native if IS_WARM else dual)(*args,**kwargs);b.record();b.synchronize();state['reference_render_ms']+=a.elapsed_time(b)
    return refpkg
def add_geometry(s):
    global GEO,COMMON,neighborpkg,meta
    i=views.index(s['viewpoint_cam']);h=hashlib.sha256();h.update(np.array([s['iteration'],i],dtype=np.int64).tobytes());h.update(s['background'].detach().cpu().numpy().tobytes())
    if not IS_WARM:h.update(s['idx'].detach().cpu().numpy().tobytes())
    sample=h.hexdigest();common.update(bytes.fromhex(sample));state.update(reference=i,neighbor=None,sample_sha=sample,common_loss=float(s['loss'].detach()),geometry_loss=0.,step_valid=0,step_kept=0)
    COMMON=s['loss'];GEO=None
    if IS_WARM:return COMMON
    j=pairs[i][0];state['neighbor']=j
    from multiview_geometry import geometry_loss
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
    with extra_stream():neighborpkg=dual(views[j],s['gaussians'],s['pipe'],s['background'],s['patch_size'],initial_stage=False)
    GEO,meta=geometry_loss(refpkg['geo_depth'],neighborpkg['geo_depth'],views[i],views[j]);b.record();b.synchronize();state['neighbor_geometry_ms']+=a.elapsed_time(b)
    kept=int(meta['mask'].sum());valid=int(meta['valid'].sum());state['geometry_calls']+=1;state['nonempty_calls']+=int(kept>0);state['valid_pixels']+=valid;state['kept_pixels']+=kept
    state.update(geometry_loss=float(GEO.detach()),step_valid=valid,step_kept=kept)
    return COMMON
def backward(s):
    global ROUTE,EXPECTED
    if IS_WARM:COMMON.backward();return
    g=s['gaussians'];ps=parameters(g);scale=next(i for i,p in enumerate(ps) if p is g._scaling);first=s['iteration']==1001
    before=fingerprint(dict(capture=g.capture(),environment=g.env_map.state_dict())) if first else None
    before_rng=fingerprint(rng_state()) if first else None;before_depth=fingerprint(s['depth_list']) if first else None
    result=gradients(COMMON,COMMON+GEO,ps,refpkg['viewspace_points'],neighborpkg['viewspace_points'],oracle=first)
    state['common_gradient_passes']+=1;state['joint_backward_passes']+=1
    state['common_gradient_ms']+=result['common_gradient_ms'];state['joint_backward_ms']+=result['joint_backward_ms']
    selected=assign_optimizer(ps,result,False);original=ROUTE['original_stats'];ROUTE=dict(result,parameter_objects=ps,selected_optimizer=selected,scale_index=scale,original_stats=original);EXPECTED=None
    if first:
        fixture,EXPECTED=statistics_fixture(g,original,refpkg['viewspace_points'],s['visibility_filter'],result['common_ref'],result['joint_ref'],result['common'][scale],result['joint'][scale])
        assert before==fingerprint(dict(capture=g.capture(),environment=g.env_map.state_dict())) and before_rng==fingerprint(rng_state()) and before_depth==fingerprint(s['depth_list'])
        put('ROUTING_FIXTURE.json',dict(passed=True,arm=LABEL,iteration=1001,new_Adam_calls=new_calls,update_geometry=False,statistics_geometry=STATISTICS,common_gradient_oracle=result['audit'],statistics_oracle=fixture,parameters_Adam_statistics_unchanged=True,global_RNG_and_depth_cache_unchanged=True,reference=state['reference'],neighbor=state['neighbor'],sample_sha=state['sample_sha'],model_state_without_gradients_sha=before,parameter_names=names(g)))
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
    if not IS_WARM and s['iteration']==15000:snapshot(s,'END_ITER_15000')
    refpkg=neighborpkg=meta=None;EXPECTED=None
    if not IS_WARM:ROUTE={'original_stats':ROUTE['original_stats']}
