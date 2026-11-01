import bpy, numpy as np, os, sys
sys.path.insert(0,'meshy'); from rlib import shoot
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/j.blend'))
def base_img(o):
    bsdf=[n for n in o.active_material.node_tree.nodes if n.type=='BSDF_PRINCIPLED'][0]
    l=bsdf.inputs['Base Color'].links[0]; n=l.from_node
    while n.type!='TEX_IMAGE': n=n.inputs[0].links[0].from_node
    return n.image
H=np.array([0.647,0.471,0.388]); B=np.array([0.529,0.427,0.380]); M=np.sqrt(H*B)
for name,g in (('cabeca',M/H),('corpo',M/B)):
    img=base_img(bpy.data.objects[name]); print(name,img.name,g.round(3))
    a=np.empty(len(img.pixels),np.float32); img.pixels.foreach_get(a); a=a.reshape(-1,4)
    a[:,:3]=np.clip(a[:,:3]*g,0,1); img.pixels.foreach_set(a.ravel()); img.update(); img.pack()
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/k.blend'))
sc=bpy.context.scene; sc.cycles.samples=40; co=sc.camera
sc.render.resolution_x,sc.render.resolution_y=600,900
for az in (0,35,90,180): shoot(co,(0,0,0.17),4.6,az,4,f'meshy/out/K_full{az}.png',lens=60)
sc.render.resolution_x,sc.render.resolution_y=900,900
shoot(co,(0,0,0.97),0.85,25,3,'meshy/out/K_face.png',lens=60)
shoot(co,(0,0,0.84),0.6,0,0,'meshy/out/K_neck.png',lens=60)
shoot(co,(0,0,0.84),0.6,90,0,'meshy/out/K_neckl.png',lens=60)
