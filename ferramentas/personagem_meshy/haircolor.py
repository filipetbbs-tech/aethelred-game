import bpy, numpy as np, os, colorsys
from PIL import Image
E=lambda k,d: float(os.environ.get(k,d))
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/fw_hm.blend'))
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.samples=1; sc.render.bake.margin=8
o=[o for o in bpy.data.objects if o.type=='MESH'][0]; m=o.active_material; N=m.node_tree.nodes; L=m.node_tree.links
for ob in bpy.data.objects: ob.select_set(ob==o)
bpy.context.view_layer.objects.active=o
# 1) máscara do cabelo no espaço da textura (bake da cor de vértice)
R=4096
mimg=bpy.data.images.new('mask_cabelo',R,R,alpha=False,float_buffer=True); mimg.colorspace_settings.name='Non-Color'
t=N.new('ShaderNodeTexImage'); t.image=mimg
for n in N: n.select=False
t.select=True; N.active=t
va=N.new('ShaderNodeVertexColor'); va.layer_name='cabelo'; em=N.new('ShaderNodeEmission'); L.new(va.outputs['Color'],em.inputs['Color'])
out=N['Material Output']; orig=out.inputs['Surface'].links[0].from_socket; L.new(em.outputs[0],out.inputs['Surface'])
bpy.ops.object.bake(type='EMIT',uv_layer=o.data.uv_layers[0].name)
L.new(orig,out.inputs['Surface']); N.remove(em); N.remove(t)
mk=np.empty(R*R*4,np.float32); mimg.pixels.foreach_get(mk); mk=mk.reshape(R,R,4)[:,:,0]
print('texels cabelo',(mk>.5).mean().round(3))
# 2) LUT de cor a partir do cabelo da arte (vista de costas)
c=np.asarray(Image.open('/tmp/claude-0/-home-user-aethelred-game/f693d26f-a336-56f4-a564-05937b894639/images/56.webp').convert('RGB').crop((1200,20,1500,470))).reshape(-1,3)/255.
mx=c.max(1); mn=c.min(1); sat=(mx-mn)/np.maximum(mx,1e-4); lumc=c@[.299,.587,.114]
c=c[(sat>.3)&(lumc>.08)]; lumc=c@[.299,.587,.114]
o_=np.argsort(lumc); c=c[o_]; NB=256
lut=np.array([np.median(b,0) for b in np.array_split(c,NB)])
print('arte cabelo: escuro',lut[10].round(3),'medio',lut[128].round(3),'claro',lut[245].round(3))
# 3) aplica: rank de luminância no cabelo original -> cor da arte
src=bpy.data.images['Image_0']; a=np.empty(R*R*4,np.float32); src.pixels.foreach_get(a); a=a.reshape(R,R,4)
dst=bpy.data.images['rosto_arte']; d=np.empty(R*R*4,np.float32); dst.pixels.foreach_get(d); d=d.reshape(R,R,4)
sel=mk>0.02
lum=a[:,:,:3]@np.array([.299,.587,.114])
lv=lum[mk>.5]; qs=np.percentile(lv,np.linspace(0,100,NB+1))
rank=np.clip(np.searchsorted(qs,lum[sel])-1,0,NB-1)
# contraste extra local para destacar mechas (opcional)
lo=E('LO',.3); newc=lut[np.clip((lo*(NB-1)+rank*(1-lo)).astype(int),0,NB-1)]
ln=(newc@np.array([.299,.587,.114]))[:,None]; newc=np.clip(ln+(newc-ln)*E('SAT',1.35),0,1)
w=np.clip(mk[sel],0,1)[:,None]
d[sel,:3]=d[sel,:3]*(1-w)+newc*w
dst.pixels.foreach_set(d.ravel()); dst.update(); dst.pack()
# 4) rugosidade: menos "plástico molhado"
rimg=bpy.data.images['Image_1']; rw,rh=rimg.size; r=np.empty(rw*rh*4,np.float32); rimg.pixels.foreach_get(r); r=r.reshape(rh,rw,4)
print('rough antes',np.percentile(r[:,:,1],[5,50,95]).round(3),'metal',np.percentile(r[:,:,2],[50,95]).round(3))
mks=np.asarray(Image.fromarray((mk*255).astype(np.uint8)).resize((rw,rh)),np.float32)/255
r[:,:,1]=np.maximum(r[:,:,1],E('RMIN',.55)*(1-mks)+E('RHAIR',.72)*mks); r[:,:,2]=0
rimg.pixels.foreach_set(r.ravel()); rimg.update(); rimg.pack()
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/fw_final2.blend'))
