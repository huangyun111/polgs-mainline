"""Differentiable copy of native local_homo, only for the new loss's VJP."""
import torch
def native_slopes(means,rot,view,settings):
 r,x,y,z=rot.unbind(-1)
 R=torch.stack((torch.stack((1-2*(y*y+z*z),2*(x*y+r*z),2*(x*z-r*y)),dim=-1),
                torch.stack((2*(x*y-r*z),1-2*(x*x+z*z),2*(y*z+r*x)),dim=-1),
                torch.stack((2*(x*z+r*y),2*(y*z-r*x),1-2*(x*x+y*y)),dim=-1)),dim=1)
 p=means@view[:3,:3]+view[3,:3]
 axes=R@view[:3,:3];u0,u1,n=axes[:,0],axes[:,1],axes[:,2]
 safe=lambda d:torch.where(abs(d)>1e-8,d,torch.ones_like(d))
 xy=p[:,:2]/safe(p[:,2:3]);one=torch.ones_like(xy[:,:1])
 d0=torch.cat((xy+torch.tensor([.001,0.],device=means.device),one),dim=-1)
 d1=torch.cat((xy+torch.tensor([0.,.001],device=means.device),one),dim=-1)
 d0=torch.nn.functional.normalize(d0,dim=-1,eps=1e-8);d1=torch.nn.functional.normalize(d1,dim=-1,eps=1e-8)
 dot=(p*n).sum(-1,keepdim=True)
 xu0=d0*(dot/safe((d0*n).sum(-1,keepdim=True)))-p
 xu1=d1*(dot/safe((d1*n).sum(-1,keepdim=True)))-p
 fx=settings.image_width/(2*settings.tanfovx);fy=settings.image_height/(2*settings.tanfovy);factor=(fx+fy)/2000
 j0=(xu0*u0).sum(-1)/factor;j1=(xu1*u0).sum(-1)/factor
 j2=(xu0*u1).sum(-1)/factor;j3=(xu1*u1).sum(-1)/factor
 return torch.stack((j0*u0[:,2]+j2*u1[:,2],j1*u0[:,2]+j3*u1[:,2]),dim=-1)
