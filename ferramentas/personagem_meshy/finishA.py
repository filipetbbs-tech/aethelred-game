import bpy, numpy as np, os
from PIL import Image
E=lambda k,d: float(os.environ.get(k,d))
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/a_proj.blend'))
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.samples=1; sc.render.bake.margin=16
W=np.array([1.0,E('WG',.9),E('WB',.82)])
def px(img):
    a=np.empty(len(img.pixels),np.float32); img.pixels.foreach_get(a); return a.reshape(img.size[1],img.size[0],4)
def put(img,a): img.pixels.foreach_set(a.ravel()); img.update(); img.pack()
# corpo: tom mais quente
body=bpy.data.objects['corpo']; bimg=[n for n in body.active_material.node_tree.nodes if n.type=='TEX_IMAGE' and n.image.name=='corpo_cor'][0].image
a=px(bimg); a[:,:,:3]*=W; put(bimg,a)
# cabeça: mesmo tom na base, depois bake
head=bpy.data.objects['cabeca']; m=head.active_material; N=m.node_tree.nodes; L=m.node_tree.links
g=[n for n in N if n.type=='MIX' and n.blend_type=='MULTIPLY'][0]; g.inputs['Factor'].default_value=1; g.inputs['B'].default_value=(*(W**2.2),1)
for ob in bpy.data.objects: ob.select_set(ob==head)
bpy.context.view_layer.objects.active=head
bsdf=N['Principled BSDF']; out=N['Material Output']; col=bsdf.inputs['Base Color'].links[0].from_socket
uvn=head.data.uv_layers[0].name
img=bpy.data.images.new('cabeca_arte',4096,4096,alpha=False)
t=N.new('ShaderNodeTexImage'); t.image=img; um=N.new('ShaderNodeUVMap'); um.uv_map=uvn; L.new(um.outputs[0],t.inputs['Vector'])
for n in N: n.select=False
t.select=True; N.active=t
em=N.new('ShaderNodeEmission'); L.new(col,em.inputs['Color']); orig=out.inputs['Surface'].links[0].from_socket; L.new(em.outputs[0],out.inputs['Surface'])
bpy.ops.object.bake(type='EMIT',uv_layer=uvn)
L.new(orig,out.inputs['Surface']); N.remove(em); L.new(t.outputs['Color'],bsdf.inputs['Base Color']); img.pack()
head.data.uv_layers.remove(head.data.uv_layers['proj'])
# cabelo: paleta da arte por ranking de luminância
c=np.asarray(Image.open('/tmp/claude-0/-home-user-aethelred-game/f693d26f-a336-56f4-a564-05937b894639/images/56.webp').convert('RGB').crop((1200,20,1500,470))).reshape(-1,3)/255.
mx=c.max(1); mn=c.min(1); sat=(mx-mn)/np.maximum(mx,1e-4); lc=c@[.299,.587,.114]
c=c[(sat>.3)&(lc>.08)]; c=c[np.argsort(c@[.299,.587,.114])]; NB=256
lut=np.array([np.median(b,0) for b in np.array_split(c,NB)])
hair=bpy.data.objects['cabelo']; hm=hair.active_material
himg=[n for n in hm.node_tree.nodes if n.type=='TEX_IMAGE' and n.image.colorspace_settings.name=='sRGB'][0].image
print('cabelo img',himg.name,himg.size[:])
a=px(himg); lum=a[:,:,:3]@np.array([.299,.587,.114]); mxa=a[:,:,:3].max(2); sa=(mxa-a[:,:,:3].min(2))/np.maximum(mxa,1e-4); sel=(lum>0.02)&(sa>E('MS',.15)); print('metal excluido',((lum>0.02)&~sel).mean().round(4))
qs=np.percentile(lum[sel],np.linspace(0,100,NB+1)); rank=np.clip(np.searchsorted(qs,lum[sel])-1,0,NB-1)
lo=E('LO',.25); newc=lut[np.clip((lo*(NB-1)+rank*(1-lo)).astype(int),0,NB-1)]
ln=(newc@np.array([.299,.587,.114]))[:,None]; newc=np.clip(ln+(newc-ln)*E('SAT',1.35),0,1)
a[sel,:3]=newc; put(himg,a)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/a_final.blend'))
