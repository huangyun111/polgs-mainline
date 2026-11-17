from pathlib import Path
import json,math,statistics
import numpy as np
R=Path('/home/hy/scap_controls_20261111');SCENE='StandingRabbit';ARMS=['Image-score','Joint-update']
def read(p):return json.loads(p.read_text())
ref=read(R/'setup/REFERENCE_REUSE.json');quality={a:x['quality'] for a,x in ref['references'].items()};counts={a:x['points'] for a,x in ref['references'].items()};capacity={};cost={};samples={}
for a in ARMS:
    w=R/'runs'/SCENE/a;t=read(w/'TRAINING_RECEIPT.json');assert t['status']=='COMPLETED' and t['adam_calls']==14999 and t['new_Adam_calls']==13999
    route=read(w/'ROUTING_FIXTURE.json');assert route['passed'] and route['update_geometry']==(a=='Joint-update') and route['statistics_geometry']
    assert t['optimizer_joint_calls']==(13999 if a=='Joint-update' else 0) and t['optimizer_common_calls']==(13999 if a=='Image-score' else 0)
    rows=[json.loads(x) for x in (w/'trace.jsonl').read_text().splitlines()];assert [x['iteration'] for x in rows]==list(range(1001,15001))
    assert all(x['forward_points']<=39000 for x in rows if x['iteration']>1100)
    samples[a]=[(x['iteration'],x['reference'],x['sample_sha']) for x in rows]
    ops=[json.loads(x) for x in (w/'budget_operations.jsonl').read_text().splitlines()];assert [x['iteration'] for x in ops]==list(range(1100,15001,100))
    for x in ops:
        c,s=x['clone_candidates'],x['split_candidates'];room=39000-x['retained'];kc=c if c+s<=room else room*c//(c+s);ks=s if c+s<=room else room-kc
        assert (x['Kc'],x['Ks'])==(kc,ks) and x['after']==x['retained']+kc+ks<=39000
        z=np.load(w/'parents'/(str(x['iteration'])+'.npz'));ranking=z['ranking']
        for name,k in [('clone',kc),('split',ks)]:
            pool=z[name+'_pool'];expected=pool[np.lexsort((pool,-ranking[pool]))[:k]]
            assert np.array_equal(expected,z[name])
            assert np.all(z['joint_score'][pool]>=1e-4)
        if a=='Image-score':assert x['ranking_signal']=='RDGS_DSSIM' and x['window_views']==list(range(20))
        else:assert np.array_equal(ranking,z['joint_score'])
    assert sum(x['forward_points'] for x in rows)==t['Gaussian_forward_steps_new']
    assert sum(x['forward_points'] for x in rows[:-1])==t['Gaussian_Adam_exposure_new']
    for key,n in [('clone_births',sum(x['Kc'] for x in ops)),('split_births',sum(2*x['Ks'] for x in ops)),('capacity_growth_limited_operations',sum(x['growth_limited'] for x in ops))]:assert t[key]==n
    if a=='Image-score':assert t['image_score_calls']==14000 and read(w/'IMAGE_SCORE_READONLY_ORACLE.json')['passed']
    q=read(w/'EVALUATION_RECEIPT.json');quality[a]=q;counts[a]=read(w/'PRE_OPERATION_15000.json')['points']
    for rep in q['replicates']:
        z=np.load(R/'evaluation'/SCENE/(SCENE+'_'+a+'_'+str(rep['seed'])+'_distances.npz'))
        for k in ['accuracy','completeness']:assert math.isclose(float(z[k+'_gt_x1000'].mean()),rep[k],rel_tol=1e-10,abs_tol=1e-10)
        assert math.isclose(rep['cd'],.5*(rep['accuracy']+rep['completeness']),rel_tol=1e-10,abs_tol=1e-10)
    for k in q['mean']:
        values=[x[k] for x in q['replicates']]
        assert math.isclose(q['mean'][k],statistics.mean(values),rel_tol=1e-9,abs_tol=1e-9)
        assert math.isclose(q['sampling_sd'][k],statistics.stdev(values),rel_tol=1e-9,abs_tol=1e-9)
    capacity[a]={k:t[k] for k in ['Gaussian_forward_steps_new','Gaussian_Adam_exposure_new','clone_births','split_births','clone_births_used','split_births_used','native_pruned','quota_extra_pruned','capacity_growth_limited_operations','opacity_cap_operations']}
    capacity[a]['forward_Gaussian_steps_total']=ref['prefix_forward_Gaussian_steps']+t['Gaussian_forward_steps_new']
    cost[a]={k:t[k] for k in ['wall_seconds','peak_reserved_bytes','gpu','pid','image_score_calls','image_error_gpu_ms','image_backproject_gpu_ms','image_score_wall_seconds','common_gradient_ms','joint_backward_ms','neighbor_geometry_ms']}
assert samples['Image-score']==samples['Joint-update']
old=Path('/home/hy/independent_cap_20261111/runs/StandingRabbit/S-cap');assert read(old/'TRAINING_RECEIPT.json')['common_sample_sha']==read(R/'runs/StandingRabbit/Image-score/TRAINING_RECEIPT.json')['common_sample_sha']
for a in ['S-cap','N-cap','S-dual']:
    capacity[a]=ref['references'][a]['capacity'];cost[a]=ref['references'][a]['cost']
comparisons={}
for a in ARMS:
    comparisons['S-cap_to_'+a]=dict(CD_percent=100*(quality[a]['mean']['cd']/quality['S-cap']['mean']['cd']-1),sampling_benefits=[x['cd']-y['cd'] for x,y in zip(quality['S-cap']['replicates'],quality[a]['replicates'])],points_percent=100*(counts[a]/counts['S-cap']-1),forward_capacity_percent=100*(capacity[a]['forward_Gaussian_steps_total']/capacity['S-cap']['forward_Gaussian_steps_total']-1))
out=dict(passed=True,quality=quality,counts=counts,capacity=capacity,cost=cost,comparisons=comparisons,all_parent_rankings_CPU_verified=True,common_sampling_matches=True,new_Adam_calls=27998,no_scene_migration=True,no_next_experiment=True)
(R/'FINAL.json').write_text(json.dumps(out,indent=2));print('STAGE2_FINAL',json.dumps(dict(CD={a:q['mean']['cd'] for a,q in quality.items()},counts=counts,comparisons=comparisons)),flush=True)
