import numpy as np
from PIL import Image
from glb import load
m=np.load('mesh.npz'); UV,I=m['UV'],m['I']; dom=np.load('dom.npy')
j,_=load('/home/user/aethelred-game/aethelred_rig_leve.glb')
names=[j['nodes'][n]['name'].replace('mixamorig','') for n in j['skins'][0]['joints']]
def grp(n):
    if n in('Head','Neck'): return 1
    if n.startswith(('Hips','Spine')) or n.endswith('Shoulder'): return 2
    if n.endswith('UpLeg'): return 3
    if n.endswith('Leg'): return 4
    if n.endswith(('Foot','ToeBase')): return 5
    if n.startswith('Tail'): return 7
    return 6
G=np.array([grp(n) for n in names])[dom]
TS=2048; lab=np.zeros((TS,TS),np.uint8)
tris=I.reshape(-1,3)
for a,b,c in tris:
    g=np.bincount(G[[a,b,c]]).argmax()
    uv=UV[[a,b,c]]*TS; xs,ys=uv[:,0],uv[:,1]
    x0,x1=max(0,int(xs.min())-1),min(TS-1,int(xs.max())+1); y0,y1=max(0,int(ys.min())-1),min(TS-1,int(ys.max())+1)
    gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
    den=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
    if abs(den)<1e-9: continue
    l0=((ys[1]-ys[2])*(gx-xs[2])+(xs[2]-xs[1])*(gy-ys[2]))/den
    l1=((ys[2]-ys[0])*(gx-xs[2])+(xs[0]-xs[2])*(gy-ys[2]))/den
    e=-1.5/np.sqrt(abs(den)+1); mk=(l0>=e)&(l1>=e)&(1-l0-l1>=e)
    sub=lab[y0:y1+1,x0:x1+1]; sub[mk&(sub==0)]=g
np.save('lab.npy',lab)
