import bpy, numpy as np, os, sys
sys.path.insert(0,'meshy'); from rlib import *
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/fw_final.blend'))
o=[o for o in bpy.data.objects if o.type=='MESH'][0]; me=o.data
nv=len(me.vertices); co=np.empty(nv*3,np.float32); me.vertices.foreach_get('co',co); co=co.reshape(-1,3)
img=bpy.data.images['Image_0']; W,H=img.size
a=np.empty(W*H*4,np.float32); img.pixels.foreach_get(a); tex=a.reshape(H,W,4)
uv=np.empty(len(me.loops)*2,np.float32); me.uv_layers[0].data.foreach_get('uv',uv); uv=uv.reshape(-1,2)
lv=np.empty(len(me.loops),np.int32); me.loops.foreach_get('vertex_index',lv)
tu=np.clip((uv[:,0]%1)*(W-1),0,W-1).astype(int); tv=np.clip((uv[:,1]%1)*(H-1),0,H-1).astype(int)
lc=tex[tv,tu,:3]; vc=np.zeros((nv,3)); cnt=np.zeros(nv); np.add.at(vc,lv,lc); np.add.at(cnt,lv,1); vc/=np.maximum(cnt,1)[:,None]
lum=vc@np.array([.299,.587,.114]); sat=(vc.max(1)-vc.min(1))/np.maximum(vc.max(1),1e-4)
print('lum pct',np.percentile(lum,[5,25,50,75,95]).round(3))
px=(co[:,0]/0.32+0.5)*800; py=(1-((co[:,2]-0.80)/0.32+0.5))*800
inner=(((px-400)/140)**2+((py-565)/190)**2<1)&(co[:,1]<0)
head=co[:,2]>float(os.environ.get('ZH','0.45'))
hair=(lum<float(os.environ.get('HL','0.40')))&(sat>float(os.environ.get('HS','0.30')))&head&~inner
print('cabelo',hair.sum(),'de',nv)
np.save('meshy/hair_v.npy',hair)
ca=me.color_attributes.get('cabelo') or me.color_attributes.new('cabelo','FLOAT_COLOR','POINT')
c=np.zeros((nv,4),np.float32); c[:,0]=hair; c[:,3]=1; ca.data.foreach_set('color',c.ravel())
# visualização: emissão vermelha onde é cabelo
m=o.active_material; N=m.node_tree.nodes; L=m.node_tree.links
va=N.new('ShaderNodeVertexColor'); va.layer_name='cabelo'; mx=N.new('ShaderNodeMix'); mx.data_type='RGBA'
bsdf=N['Principled BSDF']; src=bsdf.inputs['Base Color'].links[0].from_socket
sep=N.new('ShaderNodeSeparateColor'); L.new(va.outputs['Color'],sep.inputs[0]); L.new(sep.outputs[0],mx.inputs['Factor'])
L.new(src,mx.inputs['A']); mx.inputs['B'].default_value=(1,0,0,1); L.new(mx.outputs['Result'],bsdf.inputs['Base Color'])
co_=setup_render((500,600),8)
for az in (0,90,180): shoot(co_,(0,0,0.65),1.6,az,3,f'meshy/out/hm_{az}.png',lens=50)
L.new(src,bsdf.inputs['Base Color'])
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/fw_hm.blend'))
