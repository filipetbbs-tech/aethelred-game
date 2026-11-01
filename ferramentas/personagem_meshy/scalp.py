import bpy, bmesh, mathutils, numpy as np, os, sys
from mathutils.bvhtree import BVHTree
sys.path.insert(0,'meshy'); from rlib import shoot
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/k.blend'))
head=bpy.data.objects['cabeca']; hair=bpy.data.objects['cabelo']
bvh=BVHTree.FromObject(hair,bpy.context.evaluated_depsgraph_get())
bm=bmesh.new(); bm.from_mesh(head.data); bm.verts.ensure_lookup_table(); bm.normal_update()
# soldar vértices duplicados na costura UV só para vizinhança (sem alterar malha): usa posição
N=len(bm.verts); co=np.array([v.co[:] for v in bm.verts]); nr=np.array([v.normal[:] for v in bm.verts])
Z0=float(os.environ.get('Z0','0.95')); R=float(os.environ.get('R','0.06'))
dist_in=np.full(N,np.inf); dist_out=np.full(N,np.inf)
S,HB,BT=.23,np.array([0,.19,-.51]),np.array([0,.037,.79])
hl=(co-BT)/S+HB
mask=(hl[:,2]>0.55-0.9*(hl[:,1]+0.55))&((hl[:,2]>0.35)|(abs(hl[:,0])<0.38))&(hl[:,2]>-0.15)
cand=np.where(mask)[0]
for i in cand:
    p=mathutils.Vector(co[i]); n=mathutils.Vector(nr[i])
    h=bvh.ray_cast(p+n*1e-4,n,R)
    if h[0]: dist_out[i]=h[3]
    h=bvh.ray_cast(p-n*1e-4,-n,R*0.5)
    if h[0]: dist_in[i]=h[3]
cov=(np.isfinite(dist_out)|np.isfinite(dist_in))
MARG=float(os.environ.get('MARG','0.005'))
d=np.zeros(N)
d[np.isfinite(dist_in)]=dist_in[np.isfinite(dist_in)]+MARG
m=np.isfinite(dist_out)&~np.isfinite(dist_in); d[m]=np.maximum(0,MARG-dist_out[m])
# vizinhança por posição (une duplicatas de costura)
key={}
for i in range(N): key.setdefault(tuple(np.round(co[i],6)),[]).append(i)
nb=[set() for _ in range(N)]
for e in bm.edges:
    a,b=e.verts[0].index,e.verts[1].index; nb[a].add(b); nb[b].add(a)
for g in key.values():
    for i in g: nb[i].update(g)
# região = cobertos + dilatação (preenche risca/frestas)
reg=cov.copy()
for _ in range(int(os.environ.get('DIL','6'))):
    new=reg.copy()
    for i in np.where(reg)[0]:
        for j in nb[i]:
            if mask[j]: new[j]=True
    reg=new
base=float(os.environ.get('BASE','0.004'))
# distância em anéis até a borda da região
ring=np.where(reg,99,0)
for k in range(1,12):
    for i in np.where(ring==99)[0]:
        if any(ring[j]==k-1 for j in nb[i]): ring[i]=k
need=np.where(reg,d,0)
d=np.maximum(need,base*np.minimum(1,ring/10.0))*reg
for _ in range(int(os.environ.get('SM','8'))):
    d2=d.copy()
    for i in np.where(reg)[0]:
        d2[i]=max(need[i],np.mean([d[j] for j in nb[i]]))
    d=d2*reg
for i in np.where(d>0)[0]:
    bm.verts[i].co-=bm.verts[i].normal*d[i]
bm.to_mesh(head.data); bm.free()
print('cobertos',cov.sum(),'regiao',reg.sum(),'desloc max',d.max())
np.save('meshy/scalp_reg.npy',reg)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/l.blend'))
sc=bpy.context.scene; sc.cycles.samples=32; cam=sc.camera
sc.render.resolution_x,sc.render.resolution_y=700,700
shoot(cam,(0,0,0.97),0.85,25,3,'meshy/out/L_face.png',lens=60)
shoot(cam,(0,0,1.02),0.7,0,40,'meshy/out/L_top.png',lens=60)
shoot(cam,(0,0,1.02),0.7,150,30,'meshy/out/L_topb.png',lens=60)
