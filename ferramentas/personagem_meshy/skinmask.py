import bpy, numpy as np, os
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/l.blend'))
o=bpy.data.objects['corpo']; me=o.data
co=np.empty(len(me.vertices)*3,np.float32); me.vertices.foreach_get('co',co); co=co.reshape(-1,3)
x,y,z=np.abs(co[:,0]),co[:,1],co[:,2]
ss=lambda e0,e1,v: np.clip((v-e0)/(e1-e0),0,1)
# top (faixa): base 0.465, topo 0.605 na frente -> 0.555 atrás
ztop=0.605+(np.clip(y,-0.15,0.13)+0.15)*(0.555-0.605)/0.28
top=ss(0.455,0.47,z)*(1-ss(ztop-0.005,ztop+0.01,z))*(1-ss(0.205,0.225,x))
# calcinha: cintura 0.29 (frente) -> 0.305 (atrás); aberturas das pernas em diagonal até a virilha
zw=0.285+(np.clip(y,-0.15,0.15)+0.15)*0.02/0.3
xl=np.minimum(0.215,0.035+np.maximum(z-0.04,0))
bri=(1-ss(zw-0.005,zw+0.01,z))*ss(-0.01,0.01,z)*(1-ss(xl-0.01,xl+0.008,x))
pele=1-np.maximum(top,bri)
# densidade de sardas: forte em ombros/colo/costas altas/braços, fraca na cintura, rara nas pernas
d=0.1+0.9*ss(0.40,0.66,z)
d=np.maximum(d,np.where(x>0.22,0.75,0))      # braços
d*=pele
cols=np.zeros((len(co),4),np.float32); cols[:,0]=pele; cols[:,1]=d; cols[:,3]=1
a=me.color_attributes.get('pele') or me.color_attributes.new('pele','FLOAT_COLOR','POINT')
a.data.foreach_set('color',cols.ravel())
print('pele<0.5:',(pele<0.5).sum(),'de',len(co))
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/l2.blend'))
