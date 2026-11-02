import bpy, mathutils, math

def gpu():
    try:
        p = bpy.context.preferences.addons['cycles'].preferences
        for t in ('OPTIX', 'CUDA'):
            try:
                p.compute_device_type = t; p.get_devices()
                if any(d.type == t for d in p.devices):
                    for d in p.devices: d.use = True
                    bpy.context.scene.cycles.device = 'GPU'; return t
            except Exception: pass
    except Exception: pass
    return 'CPU'

def setup_render(res=(700, 900), samples=32):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.samples = samples; sc.cycles.use_denoising = True
    dev = gpu()
    sc.render.resolution_x, sc.render.resolution_y = res; sc.render.resolution_percentage = 100
    w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes['Background']; bg.inputs[0].default_value = (.55, .57, .6, 1); bg.inputs[1].default_value = .6
    sc.view_settings.view_transform = 'Standard'
    for n, (rot, e) in {'key': ((50, 0, 30), 3.5), 'fill': ((60, 0, -60), 1.2), 'rim': ((70, 0, 180), 2.0)}.items():
        l = bpy.data.lights.new(n, 'SUN'); l.energy = e; o = bpy.data.objects.new(n, l); sc.collection.objects.link(o)
        o.rotation_euler = [math.radians(a) for a in rot]
    cam = bpy.data.cameras.new('cam'); co = bpy.data.objects.new('cam', cam); sc.collection.objects.link(co); sc.camera = co
    print('render device', dev)
    return co

def matte(o, rough=0.6):
    # só para conferência: tira o brilho do mapa de rugosidade
    for m in o.data.materials:
        b = [n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
        for l in list(b.inputs['Roughness'].links): m.node_tree.links.remove(l)
        for l in list(b.inputs['Metallic'].links): m.node_tree.links.remove(l)
        b.inputs['Roughness'].default_value = rough; b.inputs['Metallic'].default_value = 0

def shoot(co, target, dist, az, el, path, lens=50, ortho=None):
    t = mathutils.Vector(target); a = math.radians(az); e = math.radians(el)
    co.location = t + mathutils.Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * dist
    co.rotation_euler = (t - co.location).to_track_quat('-Z', 'Y').to_euler()
    co.data.lens = lens; co.data.clip_start = 0.001 * dist; co.data.clip_end = 100 * dist
    if ortho: co.data.type = 'ORTHO'; co.data.ortho_scale = ortho
    else: co.data.type = 'PERSP'
    bpy.context.scene.render.filepath = path; bpy.ops.render.render(write_still=True)

def bbox(objs):
    pts = [o.matrix_world @ mathutils.Vector(c) for o in objs for c in o.bound_box]
    lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
    hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
    return lo, hi
