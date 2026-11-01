import bpy, os, time
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/fw_proj.blend'))
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=1; sc.render.bake.margin=16
o=[o for o in bpy.data.objects if o.type=='MESH'][0]; m=o.active_material; N=m.node_tree.nodes; L=m.node_tree.links
for ob in bpy.data.objects: ob.select_set(ob==o)
bpy.context.view_layer.objects.active=o
bsdf=N['Principled BSDF']; out=N['Material Output']
col=bsdf.inputs['Base Color'].links[0].from_socket
img=bpy.data.images.new('rosto_arte',4096,4096,alpha=False); img.colorspace_settings.name='sRGB'
t=N.new('ShaderNodeTexImage'); t.image=img
um=N.new('ShaderNodeUVMap'); um.uv_map=o.data.uv_layers[0].name; L.new(um.outputs[0],t.inputs['Vector'])
for n in N: n.select=False
t.select=True; N.active=t
em=N.new('ShaderNodeEmission'); L.new(col,em.inputs['Color']); orig=out.inputs['Surface'].links[0].from_socket
L.new(em.outputs[0],out.inputs['Surface'])
t0=time.time(); bpy.ops.object.bake(type='EMIT',uv_layer=o.data.uv_layers[0].name); print('bake',round(time.time()-t0,1))
img.filepath_raw=os.path.abspath('meshy/fw_cor.png'); img.file_format='PNG'; img.save(); img.pack()
L.new(orig,out.inputs['Surface']); N.remove(em)
L.new(t.outputs['Color'],bsdf.inputs['Base Color'])
o.data.uv_layers.remove(o.data.uv_layers['proj'])
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/fw_final.blend'))
bpy.ops.export_scene.gltf(filepath=os.path.abspath('meshy/female_warrior_rosto_arte.glb'),export_format='GLB')
