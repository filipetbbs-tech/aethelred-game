import numpy as np
from PIL import Image
def render(P,I,UV,tex,axis_u,axis_v,depth,size=512,bounds=None,mask=None):
    # simple z-buffer triangle raster with texture
    T=np.asarray(tex,np.float32); th,tw=T.shape[:2]
    u=P@axis_u; v=P@axis_v; d=P@depth
    if bounds is None: bounds=(u.min(),u.max(),v.min(),v.max())
    u0,u1,v0,v1=bounds; s=size/max(u1-u0,v1-v0)
    X=(u-u0)*s; Y=(v1-v)*s
    img=np.full((size,size,3),90,np.float32); zb=np.full((size,size),-1e9)
    tris=I.reshape(-1,3)
    if mask is not None: tris=tris[mask[tris].any(1)]
    for a,b,c in tris:
        xs=np.array([X[a],X[b],X[c]]); ys=np.array([Y[a],Y[b],Y[c]])
        x0,x1=int(max(0,np.floor(xs.min()))),int(min(size-1,np.ceil(xs.max())))
        y0,y1=int(max(0,np.floor(ys.min()))),int(min(size-1,np.ceil(ys.max())))
        if x1<x0 or y1<y0: continue
        gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
        den=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
        if abs(den)<1e-12: continue
        l0=((ys[1]-ys[2])*(gx-xs[2])+(xs[2]-xs[1])*(gy-ys[2]))/den
        l1=((ys[2]-ys[0])*(gx-xs[2])+(xs[0]-xs[2])*(gy-ys[2]))/den
        l2=1-l0-l1; m=(l0>=0)&(l1>=0)&(l2>=0)
        if not m.any(): continue
        z=l0*d[a]+l1*d[b]+l2*d[c]
        yy,xx=np.nonzero(m); yy2=yy+y0; xx2=xx+x0; zz=z[m]
        ok=zz>zb[yy2,xx2]
        if not ok.any(): continue
        yy2,xx2,zz=yy2[ok],xx2[ok],zz[ok]
        L0,L1,L2=l0[m][ok],l1[m][ok],l2[m][ok]
        tu=L0*UV[a,0]+L1*UV[b,0]+L2*UV[c,0]; tv=L0*UV[a,1]+L1*UV[b,1]+L2*UV[c,1]
        img[yy2,xx2]=T[np.clip((tv*th).astype(int),0,th-1),np.clip((tu*tw).astype(int),0,tw-1)]
        zb[yy2,xx2]=zz
    return Image.fromarray(img.astype(np.uint8))
