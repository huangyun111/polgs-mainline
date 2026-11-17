"""Full-state save/replay helpers; no optimizer updates."""
import copy,hashlib,random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def parameter_grads(g):
    return {group['name']:[None if p.grad is None else p.grad.detach().clone() for p in group['params']] for group in g.optimizer.param_groups}


def model_state(g):
    return dict(capture=copy.deepcopy(g.capture()),environment=copy.deepcopy(g.env_map.state_dict()),gradients=parameter_grads(g))


def fingerprint(obj):
    h=hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous().numpy();h.update(str((str(a.dtype),a.shape)).encode());h.update(a.tobytes())
        elif isinstance(x,np.ndarray):h.update(str((str(x.dtype),x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(repr(k).encode());walk(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for y in x:walk(y)
        else:h.update(repr(x).encode())
    walk(obj);return h.hexdigest()


def restore_model(g,state,opt):
    g.env_map.load_state_dict(state['environment'])
    g.restore(copy.deepcopy(state['capture']),opt)
    for group in g.optimizer.param_groups:
        for p,grad in zip(group['params'],state['gradients'][group['name']]):p.grad=None if grad is None else grad.clone()

