import numpy as np, json, struct, io
from PIL import Image, ImageFilter
from glb import load, acc
m=np.load('mesh.npz'); P,N,UV,I=m['P'],m['N'],m['UV'],m['I']
texI=Image.open('tex.png').convert('RGB'); tex=np.asarray(texI,np.float32); TH,TW=tex.shape[:2]
srcI=Image.open('face_src.webp').convert('RGB'); src=np.asarray(srcI,np.float32); SH,SW=src.shape[:2]
u0,v1,S=-0.09093996,0.7453922,0.0004
R=np.array([[205,190],[375,190],[290,282],[290,342],[290,462]],float)
Q=np.array([[265,400],[445,400],[345,500],[350,580],[350,690]],float)
M=np.linalg.lstsq(np.c_[R,np.ones(5)],Q,rcond=None)[0]
rx=(P[:,0]-u0)/S; ry=(v1-P[:,1])/S
IMG=np.c_[rx,ry,np.ones(len(P))]@M   # per-vertex image coords
tris=I.reshape(-1,3)
cen_r=np.c_[rx[tris].mean(1),ry[tris].mean(1)]
vn=N[tris].mean(1); cz=P[tris][:,:,2].mean(1)
ov=((cen_r[:,0]-292)/235)**2+((cen_r[:,1]-300)/300)**2
sel=(ov<1)&(vn[:,2]>0.35)&(cz>-0.12)&(P[tris][:,:,1].mean(1)>0.5)
# drop back-of-hood / tris hidden behind (z-buffer test at centroid via simple depth raster)
print('tris',sel.sum())
# raster original texture into image space (front-facing selection) -> base
base=np.zeros((SH,SW,3),np.float32); zb=np.full((SH,SW),-1e9); skin=np.zeros((SH,SW),np.float32); cov=np.zeros((SH,SW),bool)
def samp(u,v):
    x=u*TW-.5; y=v*TH-.5; x0=np.floor(x).astype(int); y0=np.floor(y).astype(int); fx=(x-x0)[:,None]; fy=(y-y0)[:,None]
    x0=np.clip(x0,0,TW-2); y0=np.clip(y0,0,TH-2)
    return (tex[y0,x0]*(1-fx)+tex[y0,x0+1]*fx)*(1-fy)+(tex[y0+1,x0]*(1-fx)+tex[y0+1,x0+1]*fx)*fy
for t in np.nonzero((ov<1.3)&(cz>-0.2)&(P[tris][:,:,1].mean(1)>0.45))[0]:
    a,b,c=tris[t]; xs=IMG[[a,b,c],0]; ys=IMG[[a,b,c],1]; d=P[[a,b,c],2]
    x0,x1=max(0,int(xs.min())),min(SW-1,int(xs.max())+1); y0,y1=max(0,int(ys.min())),min(SH-1,int(ys.max())+1)
    if x1<x0 or y1<y0: continue
    gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
    den=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
    if abs(den)<1e-9: continue
    l0=((ys[1]-ys[2])*(gx-xs[2])+(xs[2]-xs[1])*(gy-ys[2]))/den
    l1=((ys[2]-ys[0])*(gx-xs[2])+(xs[0]-xs[2])*(gy-ys[2]))/den
    l2=1-l0-l1; mk=(l0>=-.02)&(l1>=-.02)&(l2>=-.02)
    z=l0*d[0]+l1*d[1]+l2*d[2]
    yy,xx=np.nonzero(mk); Y=yy+y0; X=xx+x0; zz=z[mk]; ok=zz>zb[Y,X]
    Y,X,zz=Y[ok],X[ok],zz[ok]; L=np.stack([l0[mk][ok],l1[mk][ok],l2[mk][ok]],1)
    uv=L@UV[[a,b,c]]; col=samp(uv[:,0],uv[:,1])
    base[Y,X]=col; zb[Y,X]=zz; cov[Y,X]=True
    mx=col.max(1); mn=col.min(1); sat=(mx-mn)/(mx+1)
    skin[Y,X]=np.clip((sat-0.10)/0.08,0,1)*np.clip((mx-55)/30,0,1)
# fill holes in base by dilation
bI=Image.fromarray(base.astype(np.uint8))
for _ in range(6):
    blur=np.asarray(bI.filter(ImageFilter.MaxFilter(3)),np.float32)
    base=np.where(cov[...,None],base,blur); cov2=np.asarray(Image.fromarray((cov*255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)))>0
    cov=cov2; bI=Image.fromarray(base.astype(np.uint8))
# mask: skin (from original) * face oval feather in image space, smoothed
yy,xx=np.mgrid[0:SH,0:SW]
ovI=((xx-352)/172)**2+((yy-505)/290)**2
fe=np.clip((1-ovI)/0.4,0,1)**1.5
sk=np.asarray(Image.fromarray((skin*255).astype(np.uint8)).filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(6)),np.float32)/255
w=(fe*np.clip(sk*1.3,0,1))[...,None]
# colour-match photo to original skin tone mildly (keep photo detail)
comp=base*(1-w)+src*w
Image.fromarray(comp.clip(0,255).astype(np.uint8)).save('face_tex.png')
Image.fromarray((w[...,0]*255).astype(np.uint8)).save('face_mask.png')
np.savez('facesel.npz',sel=sel,IMG=IMG)
