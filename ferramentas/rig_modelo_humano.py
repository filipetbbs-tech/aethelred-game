"""Gera aethelred_humano_rig.glb: o modelo humano inteiro (aethelred_bracos_humanos_leve.glb, sem esqueleto) preso ao
esqueleto de aethelred_rig_leve.glb, para a forma normal do personagem (a transformação troca pelo modelo atual).
Reaproveita o alinhamento e os pesos dos braços de encaixar_bracos_humanos.py; o resto do corpo herda os pesos dos
vértices mais próximos do modelo atual (os dois já ficam sobrepostos a ~1 cm), média ponderada pela distância.
Uso: python3 ferramentas/rig_modelo_humano.py (precisa de numpy, scipy e Pillow)."""
import os
_src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'encaixar_bracos_humanos.py')).read()
exec(_src.split("z=dict(hideOld=hideOld")[0])
from scipy.spatial import cKDTree
# vértices do modelo atual que servem de fonte: sem pele/osso dos braços (as garras não batem com o braço humano)
src=~hide
tree=cKDTree(P[src]);srcI=np.nonzero(src)[0]
dd,ii=tree.query(Q2,k=6)
wk=1/np.maximum(dd,1e-4)**2;wk/=wk.sum(1,keepdims=True)
NB=len(names);acc_=np.zeros((len(Q2),NB))
for k in range(6):
    vi=srcI[ii[:,k]]
    for c in range(4):np.add.at(acc_,(np.arange(len(Q2)),J[vi,c]),W[vi,c]*wk[:,k])
# braços: pesos calculados para o braço humano (com mistura suave na borda da região)
arm=sel_all
accA=np.zeros_like(acc_)
for c in range(4):np.add.at(accA,(np.arange(len(Q2)),Jn[:,c]),Wn[:,c])
acc_[arm]=accA[arm]
# pontos fora da região do braço cujo vizinho mais próximo no modelo atual é do braço (dedos, punho, bordas):
# recebem pesos pela posição ao longo da cadeia ombro-cotovelo-punho-mão, senão ficariam presos à coxa e esticariam
allT=cKDTree(P);_,nn=allT.query(Q2)
armBones=[i for i,n in enumerate(names) if any(k in n for k in ['Arm','ForeArm','Hand']) and 'Shoulder' not in n]
aw=np.zeros(len(P))
for c in range(4):aw+=np.isin(J[:,c],armBones)*W[:,c]
extra=(~arm)&(aw[nn]>.5)
for side,sg in [('Left',1),('Right',-1)]:
    S_,E_,W_,M_=[BP[bi[side+n]] for n in ['Arm','ForeArm','Hand','HandMiddle1']]
    L1_,L2_,L3_=nrm(E_-S_),nrm(W_-E_),nrm(M_-W_)
    v=np.nonzero(extra&(Q2[:,0]*sg>0))[0]
    best=np.full(len(v),9.);pos=np.zeros(len(v));base=0
    for a_,b_,L_ in [(S_,E_,L1_),(E_,W_,L2_),(W_,M_+(M_-W_)/L3_*.08,L3_+.08)]:
        u_=(b_-a_)/nrm(b_-a_);t_=np.clip((Q2[v]-a_)@u_,0,nrm(b_-a_));r_=nrm(Q2[v]-(a_+np.outer(t_,u_)),axis=1)
        sel_=r_<best;best[sel_]=r_[sel_];pos[sel_]=base+t_[sel_];base+=nrm(b_-a_) if L_!=L3_+.08 else 0
    bl=.035
    wE=np.clip((pos-(L1_-bl))/(2*bl),0,1);wW=np.clip((pos-(L1_+L2_-bl))/(2*bl),0,1)
    ids=[bi[side+n] for n in ['Arm','ForeArm','Hand']]
    acc_[v]=0
    acc_[v,ids[0]]=(1-wE);acc_[v,ids[1]]=wE*(1-wW);acc_[v,ids[2]]=wW
print('pontos extras presos ao braço',extra.sum())
top=np.argsort(-acc_,axis=1)[:,:4];tw=np.take_along_axis(acc_,top,1);tw/=np.maximum(tw.sum(1,keepdims=True),1e-9)
JF=top.astype(np.uint8);WF=np.clip(np.round(tw*255),0,255).astype(np.uint8)
# corrige o arredondamento para a soma dar 255
WF[:,0]+=(255-WF.astype(int).sum(1)).astype(np.uint8)
print('vertices',len(Q2),'braços',arm.sum())
blobs=[];views=[];accs=[]
def add(arr,typ,ct,target=None,minmax=False,normalized=False):
    off=sum(len(b) for b in blobs);b=arr.tobytes();pad=(4-len(b)%4)%4;blobs.append(b+b'\0'*pad)
    v={'buffer':0,'byteOffset':off,'byteLength':len(b)}
    if target:v['target']=target
    views.append(v);a={'bufferView':len(views)-1,'componentType':ct,'count':len(arr),'type':typ}
    if normalized:a['normalized']=True
    if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
    accs.append(a);return len(accs)-1
UVq=np.clip(np.round(np.mod(UV2,1.0)*65535),0,65535).astype(np.uint16)
iP=add(Q2.astype(np.float32),'VEC3',5126,34962,True);iN=add(Nw.astype(np.float32),'VEC3',5126,34962)
iU=add(UVq,'VEC2',5123,34962,normalized=True);iJ=add(JF,'VEC4',5121,34962);iW=add(WF,'VEC4',5121,34962,normalized=True)
iI=add(I2.astype(np.uint32).ravel(),'SCALAR',5125,34963)
img=j2['images'][0];bv=j2['bufferViews'][img['bufferView']];jpg=d2[b02+bv.get('byteOffset',0):b02+bv.get('byteOffset',0)+bv['byteLength']]
off=sum(len(b) for b in blobs);pad=(4-len(jpg)%4)%4;blobs.append(jpg+b'\0'*pad);views.append({'buffer':0,'byteOffset':off,'byteLength':len(jpg)})
binb=b''.join(blobs)
gl={'asset':{'version':'2.0','generator':'aethelred modelo humano'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'humano','mesh':0}],
 'meshes':[{'name':'humano','primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU,'JOINTS_0':iJ,'WEIGHTS_0':iW},'indices':iI,'material':0}],
   'extras':{'inverseBind':[float(x) for x in IB.transpose(0,2,1).ravel()]}}],
 'materials':[{'name':'humano','doubleSided':True,'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':0,'roughnessFactor':.85}}],
 'textures':[{'source':0}],'images':[{'bufferView':len(views)-1,'mimeType':'image/jpeg'}],
 'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(binb)}]}
js=json.dumps(gl,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb
F=os.path.join(ROOT,'aethelred_humano_rig.glb');open(F,'wb').write(out);print(F,len(out))
