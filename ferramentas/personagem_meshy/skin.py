import bpy, os, sys
sys.path.insert(0,'meshy'); from rlib import shoot
E=lambda k,d: float(os.environ.get(k,d))
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('meshy/l2.blend'))
m=bpy.data.objects['corpo'].active_material; nt=m.node_tree; N=nt.nodes; L=nt.links
tex=N['Image Texture']; bsdf=N['Principled BSDF']
def node(t,**kw):
    n=N.new(t)
    for k,v in kw.items(): setattr(n,k,v)
    return n
tc=node('ShaderNodeTexCoord')
# máscara por região: forte em ombros/colo/costas altas, some para baixo
ca=node('ShaderNodeVertexColor',layer_name='pele'); scol=node('ShaderNodeSeparateColor'); L.new(ca.outputs['Color'],scol.inputs[0])
mz=node('ShaderNodeMath',operation='MULTIPLY'); L.new(scol.outputs[1],mz.inputs[0]); mz.inputs[1].default_value=1
# aglomerados (ruído baixa freq)
cl=node('ShaderNodeTexNoise'); L.new(tc.outputs['Object'],cl.inputs['Vector']); cl.inputs['Scale'].default_value=E('CL',9); cl.inputs['Detail'].default_value=2
clm=node('ShaderNodeMapRange'); L.new(cl.outputs['Fac'],clm.inputs[0]); clm.inputs[1].default_value=.35; clm.inputs[2].default_value=.65
dens=node('ShaderNodeMath',operation='MULTIPLY'); L.new(mz.outputs[0],dens.inputs[0]); L.new(clm.outputs[0],dens.inputs[1])
def spots(scale,thr_bias,rad):
    vo=node('ShaderNodeTexVoronoi'); vo.inputs['Scale'].default_value=scale; vo.inputs['Randomness'].default_value=1
    L.new(tc.outputs['Object'],vo.inputs['Vector'])
    sc=node('ShaderNodeSeparateColor'); L.new(vo.outputs['Color'],sc.inputs[0])
    # mantém a célula se aleatório < densidade
    keep=node('ShaderNodeMath',operation='LESS_THAN'); L.new(sc.outputs[0],keep.inputs[0])
    dm=node('ShaderNodeMath',operation='MULTIPLY'); L.new(dens.outputs[0],dm.inputs[0]); dm.inputs[1].default_value=thr_bias; L.new(dm.outputs[0],keep.inputs[1])
    # raio variável por célula
    rr=node('ShaderNodeMapRange'); L.new(sc.outputs[1],rr.inputs[0]); rr.inputs[3].default_value=rad*.45; rr.inputs[4].default_value=rad
    sp=node('ShaderNodeMath',operation='LESS_THAN'); L.new(vo.outputs['Distance'],sp.inputs[0]); L.new(rr.outputs[0],sp.inputs[1])
    # borda suave: 1 - dist/raio
    dv=node('ShaderNodeMath',operation='DIVIDE'); L.new(vo.outputs['Distance'],dv.inputs[0]); L.new(rr.outputs[0],dv.inputs[1])
    sm=node('ShaderNodeMapRange'); L.new(dv.outputs[0],sm.inputs[0]); sm.inputs[1].default_value=1; sm.inputs[2].default_value=.3
    a=node('ShaderNodeMath',operation='MULTIPLY'); L.new(keep.outputs[0],a.inputs[0]); L.new(sm.outputs[0],a.inputs[1])
    # intensidade variável por célula
    b=node('ShaderNodeMath',operation='MULTIPLY'); L.new(a.outputs[0],b.inputs[0]); L.new(sc.outputs[2],b.inputs[1])
    return b
f1=spots(E('FS',260),E('FD',.85),E('FR',.45)); f2=spots(E('FS2',520),E('FD2',.7),E('FR',.45))
fr=node('ShaderNodeMath',operation='MAXIMUM'); L.new(f1.outputs[0],fr.inputs[0]); L.new(f2.outputs[0],fr.inputs[1])
fs=node('ShaderNodeMath',operation='MULTIPLY'); L.new(fr.outputs[0],fs.inputs[0]); fs.inputs[1].default_value=E('FSTR',.75)
mix=node('ShaderNodeMix',data_type='RGBA',blend_type='MULTIPLY'); L.new(fs.outputs[0],mix.inputs['Factor'])
L.new(tex.outputs['Color'],mix.inputs['A']); mix.inputs['B'].default_value=(E('FCR',.62),E('FCG',.36),E('FCB',.25),1)
# poros: pontinhos escuros finos em todo o corpo
po=node('ShaderNodeTexVoronoi',feature='F1'); po.inputs['Scale'].default_value=E('PS',1800); L.new(tc.outputs['Object'],po.inputs['Vector'])
pm=node('ShaderNodeMapRange'); L.new(po.outputs['Distance'],pm.inputs[0]); pm.inputs[1].default_value=0; pm.inputs[2].default_value=.35; pm.inputs[3].default_value=E('PD',.86); pm.inputs[4].default_value=1
pn=node('ShaderNodeTexNoise'); L.new(tc.outputs['Object'],pn.inputs['Vector']); pn.inputs['Scale'].default_value=E('NS',400); pn.inputs['Detail'].default_value=4
pnm=node('ShaderNodeMapRange'); L.new(pn.outputs['Fac'],pnm.inputs[0]); pnm.inputs[1].default_value=.3; pnm.inputs[2].default_value=.7; pnm.inputs[3].default_value=E('NV',.93); pnm.inputs[4].default_value=1.04
pp=node('ShaderNodeMath',operation='MULTIPLY'); L.new(pm.outputs[0],pp.inputs[0]); L.new(pnm.outputs[0],pp.inputs[1])
mix2=node('ShaderNodeMix',data_type='RGBA',blend_type='MULTIPLY'); L.new(scol.outputs[0],mix2.inputs['Factor'])
L.new(mix.outputs['Result'],mix2.inputs['A'])
cr=node('ShaderNodeCombineColor'); L.new(pp.outputs[0],cr.inputs[0]); L.new(pp.outputs[0],cr.inputs[1]); L.new(pp.outputs[0],cr.inputs[2])
L.new(cr.outputs[0],mix2.inputs['B'])
L.new(mix2.outputs['Result'],bsdf.inputs['Base Color'])
# relevo dos poros no normal
bump=node('ShaderNodeBump'); bump.inputs['Strength'].default_value=E('BS',.25); bump.inputs['Distance'].default_value=0.0005
bm2=node('ShaderNodeMath',operation='MULTIPLY'); L.new(pp.outputs[0],bm2.inputs[0]); L.new(scol.outputs[0],bm2.inputs[1]); L.new(bm2.outputs[0],bump.inputs['Height']); L.new(N['Normal Map'].outputs['Normal'],bump.inputs['Normal'])
L.new(bump.outputs['Normal'],bsdf.inputs['Normal'])
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('meshy/m.blend'))
if os.environ.get('NOREND'): sys.exit()
sc=bpy.context.scene; sc.cycles.samples=32; cam=sc.camera
sc.render.resolution_x,sc.render.resolution_y=800,800
tag=os.environ.get('TAG','M')
shoot(cam,(0,0,0.84),0.6,0,0,f'meshy/out/{tag}_neck.png',lens=60)
shoot(cam,(0.18,0,0.74),0.7,30,10,f'meshy/out/{tag}_shoulder.png',lens=60)
shoot(cam,(0,0,0.45),1.6,20,5,f'meshy/out/{tag}_back.png',lens=60)
