# pinta peças específicas sobre a textura original: fivelas douradas; cinto, bolsas e botas marrom escuro
import numpy as np; from PIL import Image, ImageFilter
m=np.load('mesh.npz'); P,UV,I=m['P'],m['UV'],m['I']; dom=np.load('dom.npy')
from glb import load
j,_=load('/home/user/aethelred-game/aethelred_rig_leve.glb')
names=[j['nodes'][n]['name'].replace('mixamorig','') for n in j['skins'][0]['joints']]
legb=np.array([n.endswith(('Leg','Foot','ToeBase')) and not n.endswith('UpLeg') for n in names])[dom]
BOOT,BELT,POUCH,GOLD=1,2,3,4
def label(X,leg):
    # rótulo por ponto 3D (aplicado texel a texel, para bordas limpas)
    x,y,z=X[:,0],X[:,1],X[:,2]
    def box(x0,x1,y0,y1,z0=-9,z1=9): return (x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)&(z>=z0)&(z<=z1)
    L=np.zeros(len(X),np.uint8)
    L[leg&(y<-0.475)]=BOOT
    L[(y>=0.09)&(y<=0.175)&(np.abs(x)<0.24)&~leg]=BELT
    # faixa diagonal no peito: do broche até a axila
    A=np.array([0.01,0.375]); B=np.array([-0.19,0.235]); dv=(B-A)/np.linalg.norm(B-A)
    q=np.c_[x,y]-A; al=q@dv; pp=np.abs(q[:,0]*dv[1]-q[:,1]*dv[0])
    STRAP=False  # desligado: parece ser a borda de um meio peitoral, não uma faixa
    if STRAP: L[(al>-0.01)&(al<np.linalg.norm(B-A))&(pp<0.024)&(z>0.0)&~leg]=BELT
    L[box(0.09,0.235,-0.045,0.088,0.0)|box(-0.235,-0.09,-0.045,0.088,0.0)]=POUCH
    L[(((x-0.003)/0.036)**2+((y-0.162)/0.031)**2<1)&(z>0.08)|box(0.055,0.095,0.112,0.178,0.08)|(((x-0.023)/0.038)**2+((y-0.425)/0.04)**2<1)&(z>0.0)]=GOLD
    return L
TS=2048; L=np.zeros((TS,TS),np.uint8)
x,y=P[:,0],P[:,1]
near=(y<-0.4)|((y>0.-0.08)&(y<0.5)&(np.abs(x)<0.3))
for a,b,c in I.reshape(-1,3):
    if not near[[a,b,c]].any(): continue
    uv=UV[[a,b,c]]*TS; xs,ys=uv[:,0],uv[:,1]
    x0,x1=max(0,int(xs.min())-1),min(TS-1,int(xs.max())+1); y0,y1=max(0,int(ys.min())-1),min(TS-1,int(ys.max())+1)
    gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
    den=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
    if abs(den)<1e-9: continue
    l0=((ys[1]-ys[2])*(gx-xs[2])+(xs[2]-xs[1])*(gy-ys[2]))/den
    l1=((ys[2]-ys[0])*(gx-xs[2])+(xs[0]-xs[2])*(gy-ys[2]))/den
    e=-1.0/np.sqrt(abs(den)+1); mk=(l0>=e)&(l1>=e)&(1-l0-l1>=e)
    if not mk.any(): continue
    Lb=np.stack([l0[mk],l1[mk],1-l0[mk]-l1[mk]],1); X=Lb@P[[a,b,c]]
    leg=np.full(len(X),legb[[a,b,c]].sum()>=2)
    lab=label(X,leg); yy,xx=np.nonzero(mk); yy+=y0; xx+=x0
    L[yy,xx]=np.maximum(L[yy,xx],lab)
t=np.asarray(Image.open('tex.png').convert('RGB'),np.float32)
mx=t.max(2); r,g,b=t[...,0],t[...,1],t[...,2]
neutral=np.clip((0.17-(r-np.maximum(g,b))/(mx+10))/0.09,0,1)   # não pinta capa nem pele
PAL={BOOT:[(16,10,7),(50,32,21),(92,63,42)],BELT:[(16,10,7),(50,32,21),(92,63,42)],POUCH:[(16,10,7),(50,32,21),(92,63,42)],
     GOLD:[(38,26,8),(125,88,30),(225,185,100)]}
out=t.copy()
for k,(d,m_,l) in PAL.items():
    msk=L==k
    if not msk.any(): continue
    ref=np.percentile(mx[msk],95)
    v=np.clip(mx/ref,0,1.3); d,m_,l=map(np.array,(d,m_,l))
    c=np.where((v<.6)[...,None], d+(m_-d)*(v/.6)[...,None], m_+(l-m_)*((v-.6)/.5)[...,None])
    w=(neutral*msk); w=np.asarray(Image.fromarray((w*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)),np.float32)/255*(L>0)
    w=np.maximum(w,0)[...,None]*(L==k)[...,None]
    out=out*(1-w)+c*w
Image.fromarray(out.clip(0,255).astype(np.uint8)).save('tex_rec.png')
