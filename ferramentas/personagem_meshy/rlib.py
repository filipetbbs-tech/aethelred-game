import bpy, mathutils, math
def setup_render(res=(640,800),samples=16):
    sc=bpy.context.scene
    sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=samples
    sc.cycles.use_denoising=True
    sc.render.resolution_x,sc.render.resolution_y=res; sc.render.resolution_percentage=100
    w=bpy.data.worlds.new('w'); sc.world=w; w.use_nodes=True
    w.node_tree.nodes['Background'].inputs[0].default_value=(.55,.57,.6,1); w.node_tree.nodes['Background'].inputs[1].default_value=.6
    sc.view_settings.view_transform='Standard'
    for n,(rot,e) in {'key':((50,0,30),3.5),'fill':((60,0,-60),1.2),'rim':((110,0,180),2.0)}.items():
        l=bpy.data.lights.new(n,'SUN'); l.energy=e; o=bpy.data.objects.new(n,l); sc.collection.objects.link(o)
        o.rotation_euler=[math.radians(a) for a in rot]
    cam=bpy.data.cameras.new('cam'); co=bpy.data.objects.new('cam',cam); sc.collection.objects.link(co); sc.camera=co
    return co
def shoot(co,target,dist,az,el,path,lens=50,ortho=None):
    t=mathutils.Vector(target); a=math.radians(az); e=math.radians(el)
    co.location=t+mathutils.Vector((math.sin(a)*math.cos(e),-math.cos(a)*math.cos(e),math.sin(e)))*dist
    co.rotation_euler=(t-co.location).to_track_quat('-Z','Y').to_euler()
    co.data.lens=lens
    if ortho: co.data.type='ORTHO'; co.data.ortho_scale=ortho
    else: co.data.type='PERSP'
    bpy.context.scene.render.filepath=path; bpy.ops.render.render(write_still=True)
