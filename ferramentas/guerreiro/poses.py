# abre o GLB do jogo e fotografa o personagem em quadros das animações (deformação da pele): -- arquivo.glb prefixo
import bpy, sys, os, math, mathutils
D = os.path.dirname(__file__); sys.path.insert(0, D); from rl import *
a = sys.argv[sys.argv.index('--') + 1:]; F, TAG = a[0], a[1]
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=F)
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]; body = [o for o in bpy.data.objects if o.type == 'MESH' and o.data.vertices and len(o.data.vertices) > 1000][0]
for o in list(bpy.data.objects):
    if o.type == 'MESH' and o != body: bpy.data.objects.remove(o)
# escala para metros (o arquivo do Mixamo é minúsculo) mantendo a animação
s = 1.9 / max(body.dimensions); arm.scale = (s, s, s)
matte(body, 0.55); co = setup_render((520, 760), 24); sc = bpy.context.scene
poses = [p.split(':') for p in a[2].split(',')]
for act, fr in poses:
    arm.animation_data.action = bpy.data.actions[act]; sc.frame_set(int(fr)); bpy.context.view_layer.update()
    for az in (20, 110):
        shoot(co, (0, 0, 0.95), 4.2, az, 4, f'{D}/out2/{TAG}_{act}{fr}_{az}.png', lens=50)
print('POSES OK', [x.name for x in bpy.data.actions])
