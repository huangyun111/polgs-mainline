from pathlib import Path
import ast
E=Path(__file__).resolve().parent;P=E.parents[1]
base=(P/'GEOMETRY_UPDATE_STATS_2X2_20261109/evidence/train_routing.py').read_text(encoding='utf-8')
prefix=base.split('def snapshot(s,name):',1)[0].replace('/home/hy/geometry_update_stats_2x2_20261109','/home/hy/official_view_ns_20261109').replace("W=R/'runs'/arm","W=R/'runs'/scene/arm")
prefix+='\nclass ProbeDone(Exception):pass\n'
suffix=base.split("source=(CODE/'train.py').read_text();code=source",1)[1]
suffix="source=(CODE/'train.py').read_text();code=source"+suffix
suffix=suffix.replace('opt.iterations + 1):','_stop_iteration + 1):')
suffix=suffix.replace("*[str(x) for x in FIXED]","'15000'")
suffix=suffix.replace("'--azimuth_l','0.05'","'--azimuth_l',str(info['azimuth_l'])")
suffix=suffix.replace("_route_backward=backward_routing,","_route_backward=backward,")
suffix=suffix.replace("_idx_generator=idxgen)","_idx_generator=idxgen,_stop_iteration=1000 if IS_WARM else 15000)")
suffix=suffix.replace("    assert state['iteration']==15000 and state['adam_calls']==14999,state", "    assert (state['iteration'],state['adam_calls'])==((1000,1000) if IS_WARM else (15000,14999)),state")
suffix=suffix.replace("    assert (W/'PRE_OPERATION_15000.pth').exists() and (W/'POST_OPERATION_15000.pth').exists()", "    if not IS_WARM:assert (W/'PRE_OPERATION_15000.pth').exists() and (W/'POST_OPERATION_15000.pth').exists()")
suffix=suffix.replace("except BaseException:\n    rc=1", "except ProbeDone:\n    receipt['status']='ZERO_UPDATE_PROBE_PASSED'\nexcept BaseException:\n    rc=1")
suffix=suffix.replace("prefix_wall_seconds=prefix_wall,continuation_wall_seconds=time.monotonic()-START-prefix_wall,prefix_gpu_ms=prefix_gpu,continuation_gpu_ms=state['gpu_ms']-prefix_gpu,", "continuation_wall_seconds=time.monotonic()-START,")
code=prefix+(E/'training_hooks.py').read_text(encoding='utf-8')+'\n'+suffix
ast.parse(code);(E/'train.py').write_text(code,encoding='utf-8')
core=(E/'routing_core.py').read_text(encoding='utf-8')
core=core.replace('    raw=torch.autograd.grad(', '    ca,cb,ja,jb=[torch.cuda.Event(enable_timing=True) for _ in range(4)];ca.record()\n    raw=torch.autograd.grad(')
core=core.replace('    common=[clone(g)', '    cb.record()\n    common=[clone(g)')
core=core.replace('    joint_loss.backward(retain_graph=oracle)', '    ja.record();joint_loss.backward(retain_graph=oracle);jb.record();jb.synchronize()')
core=core.replace('return dict(common=common,joint=joint,', "return dict(common_gradient_ms=ca.elapsed_time(cb),joint_backward_ms=ja.elapsed_time(jb),common=common,joint=joint,")
ast.parse(core);(E/'routing_core.py').write_text(core,encoding='utf-8')
print('GENERATED_IDENTICAL_PATH_NS',len(code))
