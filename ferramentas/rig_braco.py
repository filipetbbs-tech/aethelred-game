"""Gera aethelred_braco_mutante_rig.glb: o braço mutante modelos/mutated_arm_claw_leve.glb (braço direito, sem esqueleto)
preso aos ossos do braço do esqueleto de aethelred_personagem.glb, nos dois lados (o esquerdo é o direito espelhado).
No arquivo o braço está deitado: ombro (corte fechado) no z negativo, mão no z positivo, palma para baixo, polegar em +x.
1) cadeia ombro-cotovelo-punho pelo centro das fatias ao longo de z; polegar = parte destacada junto à palma;
   as 4 garras = faixas laterais (x) da mão, cada uma com base nos nós e ponta no ponto mais distante;
2) o braço vai para o ombro do personagem, na escala do comprimento ombro-punho do esqueleto;
3) cada osso do braço ganha matriz de ligação própria (girado da pose do esqueleto para a do braço, extras.inverseBind)
   e os pesos saem da distância aos ossos, suavizados pela superfície.
Rodar depois de ferramentas/montar_personagem.py. Uso: python3 ferramentas/rig_braco.py"""
import os,json,struct,io,base64,numpy as np
import scipy.sparse as sp
from scipy.cluster.vq import kmeans2
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nrm=np.linalg.norm
def loadglb(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b0=20+l+8
    def acc(i):
        a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=b0+bv.get('byteOffset',0)+a.get('byteOffset',0)
        ct={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        es=np.dtype(ct).itemsize*n;st=bv.get('byteStride',es)
        if st==es:return np.frombuffer(d,dtype=ct,count=a['count']*n,offset=off).reshape(a['count'],n)
        raw=np.frombuffer(d,dtype=np.uint8,count=st*(a['count']-1)+es,offset=off)
        return np.lib.stride_tricks.as_strided(raw,shape=(a['count'],es),strides=(st,1)).copy().view(ct).reshape(a['count'],n)
    return d,j,b0,acc
def imgbytes(d,j,b0,k):
    img=j['images'][k];bv=j['bufferViews'][img['bufferView']];o=b0+bv.get('byteOffset',0);return d[o:o+bv['byteLength']],img.get('mimeType','image/jpeg')
def rot_between(a,b_):
    a=a/nrm(a);b_=b_/nrm(b_);v=np.cross(a,b_);c=a@b_
    if nrm(v)<1e-9:return np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);return np.eye(3)+vx+vx@vx*(1/(1+c))
clean=lambda s:s.replace('mixamorig:','').replace('mixamorig','')
# --- esqueleto do personagem ---
d,j,b0,acc=loadglb(os.path.join(ROOT,'aethelred_personagem.glb'))
sk=j['skins'][0];names=[clean(j['nodes'][i]['name']) for i in sk['joints']];bi={n:i for i,n in enumerate(names)}
IB=acc(sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1).astype(np.float64);BW=np.linalg.inv(IB);BP={n:BW[bi[n]][:3,3] for n in names}
PBm=acc(j['meshes'][0]['primitives'][0]['attributes']['POSITION']);U=(PBm[:,1].max()-PBm[:,1].min())/1.8   # unidade do espaço da malha
# --- braço (arquivo) ---
dA,jA,bA,accA=loadglb(os.path.join(ROOT,'modelos','mutated_arm_claw_leve.glb'));pA=jA['meshes'][0]['primitives'][0]
X=accA(pA['attributes']['POSITION']).astype(np.float64);NA=accA(pA['attributes']['NORMAL']).astype(np.float64)
UVA=accA(pA['attributes']['TEXCOORD_0']).astype(np.float32);IA=accA(pA['indices']).reshape(-1,3).astype(int)
def cent(z,w=.04):
    m=np.abs(X[:,2]-z)<w;return X[m].mean(0)
Sa=cent(-1.0);Wa=cent(.18)
zs=np.linspace(-1.0,.18,30);cl=np.array([cent(z) for z in zs]);seg=nrm(np.diff(cl,axis=0),axis=1);arc=np.concatenate([[0],np.cumsum(seg)])
Ea=cl[np.argmin(np.abs(arc-arc[-1]*.486))];Ha=cent(.42,.06);dH=nrm(X-Ha,axis=1)
thumb=(dH>.18)&(X[:,0]>.3)&(X[:,2]<.66)&(X[:,1]<.45);tp=X[thumb]
FA={'Thumb':(tp[np.argmin(nrm(tp-Ha,axis=1))],tp[np.argmax(nrm(tp-Ha,axis=1))])}
fing=(X[:,2]>.5)&~thumb&(dH>.12);far=fing&(dH>.38)
cx,lab=kmeans2(X[far][:,0:1],4,minit='++',seed=5);order=np.argsort(-cx[:,0])
for fi,f in enumerate(['Index','Middle','Ring','Pinky']):
    p=X[far][lab==order[fi]];tip=p[np.argmax(nrm(p-Ha,axis=1))]
    band=fing&(np.abs(X[:,0]-cx[order[fi],0])<.06)&(dH<.3)
    FA[f]=(X[band].mean(0) if band.sum()>20 else Ha+(tip-Ha)*.35,tip)
# --- arquivo -> espaço da malha do personagem (braço direito: +z do arquivo vira -x, polegar +x vira frente +z) ---
Rm=np.array([[0,0,-1],[0,1,0],[1,0,0]],float)
Sb,Eb,Wb=BP['RightArm'],BP['RightForeArm'],BP['RightHand']
s_=(nrm(Eb-Sb)+nrm(Wb-Eb))/(nrm(Ea-Sa)+nrm(Wa-Ea))
toB=lambda P_:(np.asarray(P_)-Sa)@Rm.T*s_+Sb
NJ={'RightArm':toB(Sa),'RightForeArm':toB(Ea),'RightHand':toB(Wa)}
for f,(bb,tt) in FA.items():
    for kk,fr in zip((1,2,3),(0,.38,.68)):NJ[f'RightHand{f}{kk}']=toB(bb+(tt-bb)*fr)
    NJ[f'RightHand{f}4']=toB(tt)
mir=np.array([-1,1,1.])
for n in list(NJ):NJ['Left'+n[5:]]=NJ[n]*mir
def child(n):
    if n.endswith('Arm') and 'Fore' not in n:return n.replace('Arm','ForeArm')
    if n.endswith('ForeArm'):return n.replace('ForeArm','Hand')
    if n.endswith('Hand'):return n+'Middle1'
    return n[:-1]+str(int(n[-1])+1)
IBa=IB.copy();armB=[]
for side in ['Right','Left']:
    for n in [side+'Arm',side+'ForeArm',side+'Hand']+[f'{side}Hand{f}{k}' for f in FA for k in (1,2,3)]:
        if n not in bi:continue
        c=child(n);do=(BP[c]-BP[n]) if c in BP else (BP[n]-BP[n[:-1]+str(int(n[-1])-1)])
        dn=NJ[c]-NJ[n];M=np.eye(4);M[:3,:3]=rot_between(do,dn)@BW[bi[n]][:3,:3];M[:3,3]=NJ[n];IBa[bi[n]]=np.linalg.inv(M)
        if side=='Right':armB.append(n)
print('escala do braço',round(s_/U,3),'(em unidades do corpo de 1,8 m)')
# --- pesos (braço direito) ---
Xr=toB(X);Nr=NA@Rm.T
segs=[(n,NJ[n],NJ[child(n)]) for n in armB]+[('RightShoulder',BP['RightShoulder'],NJ['RightArm'])]
D=np.stack([nrm(Xr-(a+np.outer(np.clip(((Xr-a)@(e-a))/max((e-a)@(e-a),1e-20),0,1),e-a)),axis=1) for _,a,e in segs],1)
dmin=D.min(1,keepdims=True);Wt=np.where(D<dmin*1.35+.004*U,1/np.maximum(D,1e-3*U)**4,0);Wt/=Wt.sum(1,keepdims=True)
q=np.round(X/1e-5).astype(np.int64);_,inv=np.unique(q,axis=0,return_inverse=True);inv=inv.ravel();Tw=inv[IA];nq=inv.max()+1
A=sp.coo_matrix((np.ones(Tw.size*2),(Tw[:,[0,1,2,1,2,0]].ravel(),Tw[:,[1,2,0,0,1,2]].ravel())),shape=(nq,nq)).tocsr();dg=np.asarray(A.sum(1)).ravel()+1
Wq=np.zeros((nq,Wt.shape[1]));cn=np.zeros(nq);np.add.at(Wq,inv,Wt);np.add.at(cn,inv,1);Wq/=cn[:,None]
hand=np.zeros(nq);np.add.at(hand,inv,(nrm(Xr-NJ['RightHand'],axis=1)<nrm(NJ['RightHand']-NJ['RightForeArm'])*1.6).astype(float));hand=hand>0
for _ in range(4):S=(A@Wq+Wq)/dg[:,None];Wq=np.where(hand[:,None],Wq*.8+S*.2,S)
Wt=Wq[inv];top=np.argsort(-Wt,axis=1)[:,:4];tw=np.take_along_axis(Wt,top,1);tw/=np.maximum(tw.sum(1,keepdims=True),1e-9)
topR=np.array([bi[segs[t][0]] for t in top.ravel()]).reshape(top.shape)
topL=np.array([bi['Left'+names[t][5:]] if names[t].startswith('Right') else t for t in topR.ravel()]).reshape(topR.shape)
nv=len(X);Xall=np.vstack([Xr,Xr*mir]).astype(np.float32);Nall=np.vstack([Nr,Nr*mir]).astype(np.float32);UV=np.vstack([UVA,UVA])
Tall=np.vstack([IA,IA[:,[0,2,1]]+nv]).astype(np.uint32)
JF=np.vstack([topR,topL]).astype(np.uint8);WF=np.clip(np.round(np.vstack([tw,tw])*255),0,255).astype(np.uint8);WF[:,0]+=(255-WF.astype(int).sum(1)).astype(np.uint8)
# --- grava ---
blobs=[];views=[];accs=[]
def add(arr,typ,ct,target=None,minmax=False,normalized=False):
    off=sum(len(x) for x in blobs);bb=arr.tobytes();blobs.append(bb+b'\0'*((4-len(bb)%4)%4))
    vv={'buffer':0,'byteOffset':off,'byteLength':len(bb)}
    if target:vv['target']=target
    views.append(vv);a={'bufferView':len(views)-1,'componentType':ct,'count':len(arr),'type':typ}
    if normalized:a['normalized']=True
    if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
    accs.append(a);return len(accs)-1
iP=add(Xall,'VEC3',5126,34962,True);iN=add(Nall,'VEC3',5126,34962);iU=add(UV,'VEC2',5126,34962)
iJ=add(JF,'VEC4',5121,34962);iW=add(WF,'VEC4',5121,34962,normalized=True);iI=add(Tall.ravel(),'SCALAR',5125,34963)
images=[]
for k in range(len(jA['images'])):
    bb,mt=imgbytes(dA,jA,bA,k);off=sum(len(x) for x in blobs);blobs.append(bb+b'\0'*((4-len(bb)%4)%4));views.append({'buffer':0,'byteOffset':off,'byteLength':len(bb)})
    images.append({'bufferView':len(views)-1,'mimeType':mt})
binb=b''.join(blobs)
gl={'asset':{'version':'2.0','generator':'aethelred rig braco'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'bracos','mesh':0}],
 'meshes':[{'name':'bracos','primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU,'JOINTS_0':iJ,'WEIGHTS_0':iW},'indices':iI,'material':0}],
   'extras':{'inverseBind':[float(x) for x in IBa.transpose(0,2,1).ravel()]}}],
 'materials':[jA['materials'][0]],'textures':[{'source':t['source']} for t in jA['textures']],'images':images,'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(binb)}]}
js=json.dumps(gl,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb
F=os.path.join(ROOT,'aethelred_braco_mutante_rig.glb');open(F,'wb').write(out);print(F,len(out))
