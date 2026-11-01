import bpy, os, time
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/m.blend'))
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.device='CPU'
o=bpy.data.objects['corpo']; m=o.active_material; nt=m.node_tree; N=nt.nodes; L=nt.links
bsdf=N['Principled BSDF']; out=N['Material Output']
for ob in bpy.data.objects: ob.select_set(ob==o)
bpy.context.view_layer.objects.active=o
R=int(os.environ.get('RES','4096'))
def target(name,cs):
    img=bpy.data.images.new(name,R,R,alpha=False,float_buffer=False); img.colorspace_settings.name=cs
    t=N.new('ShaderNodeTexImage'); t.image=img
    for n in N: n.select=False
    t.select=True; N.active=t; return img,t
# 1) cor: emissão da cor final
colsock=bsdf.inputs['Base Color'].links[0].from_socket
img,t=target('corpo_cor','sRGB')
em=N.new('ShaderNodeEmission'); L.new(colsock,em.inputs['Color'])
orig=out.inputs['Surface'].links[0].from_socket
L.new(em.outputs[0],out.inputs['Surface'])
sc.cycles.samples=1; sc.render.bake.margin=16
t0=time.time(); bpy.ops.object.bake(type='EMIT'); print('cor',time.time()-t0)
img.filepath_raw=os.path.abspath('meshy/corpo_cor.png'); img.file_format='PNG'; img.save()
L.new(orig,out.inputs['Surface']); N.remove(em)
# 2) normal (normal map original + relevo dos poros)
nimg,nt2=target('corpo_normal','Non-Color')
sc.cycles.samples=int(os.environ.get('NS','4'))
t0=time.time(); bpy.ops.object.bake(type='NORMAL',normal_space='TANGENT'); print('normal',time.time()-t0)
nimg.filepath_raw=os.path.abspath('meshy/corpo_normal.png'); nimg.file_format='PNG'; nimg.save()
# troca os nós: cor e normal assados
L.new(t.outputs['Color'],bsdf.inputs['Base Color'])
L.new(nt2.outputs['Color'],N['Normal Map'].inputs['Color'])
L.new(N['Normal Map'].outputs['Normal'],bsdf.inputs['Normal'])
img.pack(); nimg.pack()
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/n.blend'))
