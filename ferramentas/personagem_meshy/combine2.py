import bpy, bmesh, sys, mathutils, os
sys.path.insert(0,'meshy'); from rlib import *
U='/root/.claude/uploads/f693d26f-a336-56f4-a564-05937b894639/'
E=lambda k,d: float(os.environ.get(k,d))
S=E('S',.23); CUT=E('CUT',-.62); TILT=E('TILT',.4)
HB=mathutils.Vector((0,E('HY',.19),E('HZ',-.51))); BT=mathutils.Vector((0,E('BY',.037),E('BZ',.79)))
K=mathutils.Vector((E('KX',1.25),E('KY',1.25),E('KZ',1.25))); HO=mathutils.Vector((0,E('RY',.03),E('RZ',.03)))
def imp(f,name):
    n=set(bpy.data.objects); bpy.ops.import_scene.gltf(filepath=U+f)
    o=[o for o in set(bpy.data.objects)-n if o.type=='MESH'][0]; o.name=name
    mw=o.matrix_world.copy(); o.parent=None; o.data.transform(mw); o.matrix_world=mathutils.Matrix(); return o
bpy.ops.wm.read_factory_settings(use_empty=True)
body=imp('38ce7cb6-Meshy_AI_Realistic_Female_Body_1101131059_image-to-3d-texture.glb','corpo')
head=imp('b85da525-Meshy_AI_Realistic_Head_Decima_1101124733_texture.glb','cabeca')
hair=imp('9fbe98a4-Meshy_AI_Braided_Hair_GameRead_1101131542_texture.glb','cabelo')
bm=bmesh.new(); bm.from_mesh(head.data)
bmesh.ops.delete(bm,geom=[v for v in bm.verts if v.co.z<CUT+TILT*(v.co.y-HB.y)],context='VERTS')
bm.to_mesh(head.data); bm.free()
# cabelo -> unidades da cabeça: escala por eixo, topo alinhado ao topo do crânio
hair.data.transform(mathutils.Matrix.Diagonal((K.x,K.y,K.z,1)))
top=max(v.co.z for v in hair.data.vertices)
hair.data.transform(mathutils.Matrix.Translation(mathutils.Vector((0,0,.951-top))+HO))
M=mathutils.Matrix.Translation(BT)@mathutils.Matrix.Scale(S,4)@mathutils.Matrix.Translation(-HB)
head.data.transform(M); hair.data.transform(M)
for o in list(bpy.data.objects):
    if o.type!='MESH': bpy.data.objects.remove(o)
co=setup_render((480,600),12); tag=os.environ.get('TAG','g')
hz=BT.z+(0.25-HB.z)*S
for az in (0,90,180,270): shoot(co,(0,0,hz),1.4,az,4,f'meshy/out/{tag}_head{az}.png')
if os.environ.get('FULL'):
  for az in (0,90): shoot(co,(0,0,.1),5.2,az,3,f'meshy/out/{tag}_full{az}.png')
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(f'meshy/{tag}.blend'))
