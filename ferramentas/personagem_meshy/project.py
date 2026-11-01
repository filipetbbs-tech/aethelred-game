import bpy, numpy as np, os, colorsys, time
E=lambda k,d: float(os.environ.get(k,d))
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/fw.blend'))
o=[o for o in bpy.data.objects if o.type=='MESH'][0]; me=o.data
OS,CZ,R=0.32,0.80,800
nv=len(me.vertices)
co=np.empty(nv*3,np.float32); me.vertices.foreach_get('co',co); co=co.reshape(-1,3)
me.calc_normals_split() if hasattr(me,'calc_normals_split') else None
nr=np.empty(nv*3,np.float32); me.vertices.foreach_get('normal',nr); nr=nr.reshape(-1,3)
px=(co[:,0]/OS+0.5)*R; py=(1-((co[:,2]-CZ)/OS+0.5))*R
ss=lambda e0,e1,v: np.clip((v-e0)/(e1-e0),0,1)
facing=ss(E('F0',.30),E('F1',.65),-nr[:,1])
front=(co[:,1]<0.05)
def ell(cx,cy,ax,ay,fe):
    d=np.sqrt(((px-cx)/ax)**2+((py-cy)/ay)**2); return 1-ss(1-fe,1,d)
outer=ell(400,545,E('AX',175),E('AY',235),0.18)
inner=ell(400,565,140,190,0.0)>0.5
# cor original por vértice (média dos loops) para detectar cabelo do modelo
img=bpy.data.images['Image_0']; W,H=img.size
a=np.empty(W*H*4,np.float32); img.pixels.foreach_get(a); tex=a.reshape(H,W,4)
uv=np.empty(len(me.loops)*2,np.float32); me.uv_layers.active.data.foreach_get('uv',uv); uv=uv.reshape(-1,2)
lv=np.empty(len(me.loops),np.int32); me.loops.foreach_get('vertex_index',lv)
tu=np.clip((uv[:,0]%1)*(W-1),0,W-1).astype(int); tv=np.clip((uv[:,1]%1)*(H-1),0,H-1).astype(int)
lc=tex[tv,tu,:3]; vc=np.zeros((nv,3)); cnt=np.zeros(nv)
np.add.at(vc,lv,lc); np.add.at(cnt,lv,1); vc/=np.maximum(cnt,1)[:,None]
lum=vc@np.array([.299,.587,.114])
hair=(lum<E('HL',.2))&~inner
mask=facing*outer*front*(~hair)
print('mask>0.5:',(mask>.5).sum(),'facing',(facing>.5).sum(),'outer',(outer>.5).sum(),'front',front.sum(),'hair',hair.sum(),'inner',inner.sum(),'combo',((facing>.5)&(outer>.5)).sum()); print('lum skin', np.percentile(lum[inner],[10,50,90]).round(3))
# UV de projeção
pl=np.stack([px[lv]/R,1-py[lv]/R],1).astype(np.float32)
puv=me.uv_layers.get('proj') or me.uv_layers.new(name='proj')
puv.data.foreach_set('uv',pl.ravel())
for l in me.uv_layers: l.active_render=(l.name!='proj')
me.uv_layers.active=me.uv_layers[0]
ca=me.color_attributes.get('mask') or me.color_attributes.new('mask','FLOAT_COLOR','POINT')
cols=np.zeros((nv,4),np.float32); cols[:,:3]=mask[:,None]; cols[:,3]=1; ca.data.foreach_set('color',cols.ravel())
# ganho de cor na faixa da borda (arte -> pele do modelo), aplicado só perto da borda
cimg=np.asarray(__import__('PIL.Image',fromlist=['x']).open('meshy/concept_warp.png').convert('RGB')).astype(np.float32)/255
band=(mask>.85)
cx_=np.clip(px[band].astype(int),0,R-1); cy_=np.clip(py[band].astype(int),0,R-1)
cc=np.median(cimg[cy_,cx_]**2.2,0)*E('TB',1.25); oc=np.median(vc[band],0); gain=cc/oc
print('arte',cc.round(3),'modelo',oc.round(3),'ganho',gain.round(3))
# material: mistura
m=o.active_material; nt=m.node_tree; N=nt.nodes; L=nt.links
bsdf=N['Principled BSDF']; otex=bsdf.inputs['Base Color'].links[0].from_node
ci=N.new('ShaderNodeTexImage'); ci.image=bpy.data.images.load(os.path.abspath('meshy/concept_warp.png')); ci.extension='EXTEND'
um=N.new('ShaderNodeUVMap'); um.uv_map='proj'; L.new(um.outputs[0],ci.inputs['Vector'])
va=N.new('ShaderNodeVertexColor'); va.layer_name='mask'
sep=N.new('ShaderNodeSeparateColor'); L.new(va.outputs['Color'],sep.inputs[0])
# correção de borda: fator = 1 - máscara^k  (borda corrigida, centro fiel à arte)
g=N.new('ShaderNodeMix'); g.data_type='RGBA'; g.blend_type='MULTIPLY'; g.inputs['Factor'].default_value=E('GB',1.0)
L.new(otex.outputs['Color'],g.inputs['A']); lg=gain; g.inputs['B'].default_value=(lg[0],lg[1],lg[2],1)
mx=N.new('ShaderNodeMix'); mx.data_type='RGBA'; L.new(sep.outputs[0],mx.inputs['Factor'])
L.new(g.outputs['Result'],mx.inputs['A']); L.new(ci.outputs['Color'],mx.inputs['B'])
L.new(mx.outputs['Result'],bsdf.inputs['Base Color'])
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/fw_proj.blend'))
