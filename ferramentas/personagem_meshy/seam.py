import bpy, bmesh, mathutils, numpy as np, os
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/i1.blend'))
body=bpy.data.objects['corpo']; head=bpy.data.objects['cabeca']
S,CUT,TILT,HZ,BZ,BY=.23,-.67,.5,-.51,.79,.037
plane=lambda y: BZ+(CUT-HZ)*S+TILT*(y-BY)
# 1) remove o toco do pescoço do corpo que fica dentro do pescoço da cabeça
bm=bmesh.new(); bm.from_mesh(body.data)
dl=[v for v in bm.verts if v.co.z>plane(v.co.y)+0.03 and abs(v.co.x)<0.12 and v.co.z>0.70]
print('corpo removidos',len(dl)); bmesh.ops.delete(bm,geom=dl,context='VERTS'); bm.to_mesh(body.data); bm.free()
bvh=BVHTree.FromObject(body,bpy.context.evaluated_depsgraph_get())
# 2) anel do corte da cabeça = maior laço de borda
bm=bmesh.new(); bm.from_mesh(head.data); bm.verts.ensure_lookup_table()
bverts={v for v in bm.verts if any(e.is_boundary for e in v.link_edges)}
ring=[v for v in bverts if -0.002<=v.co.z-plane(v.co.y)<0.008]
print('anel',len(ring),min(v.co.z for v in ring),max(v.co.z for v in ring))
kd=KDTree(len(ring))
for i,v in enumerate(ring): kd.insert(v.co,i)
kd.balance()
BAND=0.03; moved=0
for v in bm.verts:
    p,i,d=kd.find(v.co)
    if d>BAND: continue
    loc,nor,_,dist=bvh.find_nearest(v.co)
    if loc is None or dist>0.05: continue
    w=(1-d/BAND)**2
    if (v.co-loc).dot(nor)>-0.002:
        v.co=v.co.lerp(loc-nor*0.002,w); moved+=1
bm.to_mesh(head.data); bm.free(); print('movidos',moved)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/j.blend'))
