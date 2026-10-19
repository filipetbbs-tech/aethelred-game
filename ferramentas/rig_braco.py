"""Gera aethelred_braco_mutante_rig.glb: o braço mutante modelos/mutated_arm_claw_leve.glb (braço direito, sem esqueleto)
preso aos ossos do braço do esqueleto, nos dois lados (o esquerdo é o direito espelhado).
No arquivo o braço está deitado: ombro (corte fechado) no z negativo, mão no z positivo, palma para baixo, polegar em +x.
1) cadeia ombro-cotovelo-punho pelo centro das fatias ao longo de z; polegar = parte destacada junto à palma;
   as 4 garras = faixas laterais (x) da mão, cada uma com base nos nós e ponta no ponto mais distante;
2) o braço vai para o ombro do corpo (aethelred_corpo_rig.glb), na escala do comprimento ombro-punho do corpo;
3) matrizes de ligação próprias de cada osso do braço (extras.inverseBind) e pesos pela distância aos ossos.
Rodar depois de ferramentas/rig_tpose.py. Uso: python3 ferramentas/rig_braco.py"""
import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__name_bak=__name__;__name__='rig_braco_lib'
exec(open(os.path.join(ROOT,'ferramentas','rig_tpose.py')).read())
__name__=__name_bak
from scipy.cluster.vq import kmeans2
A=load_static(os.path.join(ROOT,'modelos','mutated_arm_claw_leve.glb'))
X=A['P']
def cent(z,w=.04):
    m=np.abs(X[:,2]-z)<w;return X[m].mean(0)
Sa=cent(-1.0);Wa=cent(.18)
zs=np.linspace(-1.0,.18,30);cl=np.array([cent(z) for z in zs]);seg=nrm(np.diff(cl,axis=0),axis=1);arc=np.concatenate([[0],np.cumsum(seg)])
Ea=cl[np.argmin(np.abs(arc-arc[-1]*.486))]
Ha=cent(.42,.06)   # palma
d=nrm(X-Ha,axis=1)
thumb=(d>.18)&(X[:,0]>.3)&(X[:,2]<.66)&(X[:,1]<.45)
tp=X[thumb];t_tip=tp[np.argmax(nrm(tp-Ha,axis=1))];t_base=tp[np.argmin(nrm(tp-Ha,axis=1))]
fing=(X[:,2]>.5)&~thumb&(d>.12)
far=fing&(d>.38)
cx,lab=kmeans2(X[far][:,0:1],4,minit='++',seed=5);order=np.argsort(-cx[:,0])   # +x = lado do polegar = indicador
FA={'Thumb':(t_base,t_tip)}
for fi,f in enumerate(['Index','Middle','Ring','Pinky']):
    p=X[far][lab==order[fi]];tip=p[np.argmax(nrm(p-Ha,axis=1))]
    band=fing&(np.abs(X[:,0]-cx[order[fi],0])<.06)&(d<.3)
    base=X[band].mean(0) if band.sum()>20 else Ha+(tip-Ha)*.35
    FA[f]=(base,tip)
print('braço (arquivo): ombro',Sa.round(3),'cotovelo',Ea.round(3),'punho',Wa.round(3))
for f,(b_,t_) in FA.items():print(' ',f,'base',b_.round(3),'ponta',t_.round(3))
# --- corpo: juntas e ligação ---
B=load_static(os.path.join(ROOT,'modelos','rogue_warrior_t_pose_leve.glb'))
ti=B['j']['textures'][B['mat']['pbrMetallicRoughness']['baseColorTexture']['index']]['source']
C=sample(np.asarray(Image.open(io.BytesIO(imgbytes(B['d'],B['j'],B['b0'],ti)[0])).convert('RGB')).astype(np.float64)/255,B['UV'])
NJ,k=fit_joints(B,C)
# arquivo -> corpo (braço direito: eixo +z do arquivo vira -x do corpo, polegar +x vira frente +z)
Rm=np.array([[0,0,-1],[0,1,0],[1,0,0]],float)
Sb,Eb,Wb=NJ['RightArm'],NJ['RightForeArm'],NJ['RightHand']
s_=(nrm(Eb-Sb)+nrm(Wb-Eb))/(nrm(Ea-Sa)+nrm(Wa-Ea))
toB=lambda P_:(np.asarray(P_)-Sa)@Rm.T*s_+Sb
print('escala do braço',round(s_,3))
NJr=dict(NJ)
NJr['RightArm']=toB(Sa);NJr['RightForeArm']=toB(Ea);NJr['RightHand']=toB(Wa)
mid=FA['Middle'][0]
NJr['RightHandMiddle1']=toB(mid)
for f,(b_,t_) in FA.items():
    for kk,fr in zip((1,2,3),(0,.38,.68)):NJr[f'RightHand{f}{kk}']=toB(b_+(t_-b_)*fr)
    NJr[f'RightHand{f}_tip']=toB(t_)
# lado esquerdo: espelho em x (corpo)
mir=np.array([-1,1,1.])
for n in list(NJr.keys()):
    if n.startswith('Right') and ('Arm' in n or 'Hand' in n):NJr['Left'+n[5:]]=NJr[n]*mir
IBa,JO,toOld=build_bind(NJr,k)
# --- malha: direita + esquerda espelhada ---
Xr=toB(X);Nr=A['N']@Rm.T;Xl=Xr*mir;Nl=Nr*mir
armB=[n for n in names if n.startswith('Right') and ('Arm' in n or 'Hand' in n) and 'Shoulder' not in n]
segs=segments(NJr,armB+['RightShoulder'])
handMask=nrm(Xr-NJr['RightHand'],axis=1)<nrm(NJr['RightHand']-NJr['RightForeArm'])*1.6
top,tw=skin_weights(Xr,A['I'],segs,handMask,it=4)
topR=np.array([bi[segs[t][0]] for t in top.ravel()]).reshape(top.shape)
topL=np.array([bi['Left'+names[t][5:]] if names[t].startswith('Right') else t for t in topR.ravel()]).reshape(topR.shape)
nv=len(X)
Xall=np.vstack([Xr,Xl]);Nall=np.vstack([Nr,Nl]);UV=np.vstack([A['UV'],A['UV']])
Tall=np.vstack([A['I'],A['I'][:,[0,2,1]]+nv])
JF,WF=wquant(np.vstack([topR,topL]),np.vstack([tw,tw]))
write_glb(os.path.join(ROOT,'aethelred_braco_mutante_rig.glb'),toOld(Xall),Nall@FO.T,UV,Tall,JF,WF,IBa,A)
