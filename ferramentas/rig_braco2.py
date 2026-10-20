"""Gera aethelred_braco_mutante_rig.glb a partir do braço novo (modelos/mutated_arm_shoulder_leve.glb: braço direito com ombro,
peito e costas, sem esqueleto), preso ao esqueleto de aethelred_personagem.glb; o esquerdo é o direito espelhado.
No arquivo: ombro em z negativo, mão em z positivo, palma para baixo, polegar em +x (o braço é um pouco diagonal).
1) juntas do braço: ombro, cotovelo e pulso pelo centro das fatias; cada osso ganha matriz de ligação própria;
2) pesos por distância aos ossos (braço, antebraço, mão, ombro e coluna: o pedaço de peito/costas segue o tronco),
   suavizados pela superfície;
3) dedos: as 5 pontas das garras são os pontos mais distantes da palma pela malha; cada vértice da mão fica com o dedo
   da ponta mais próxima (pela malha) e ganha 3 juntas próprias, no caminho do dedo (como ferramentas/dedos_mutantes.py).
Uso: python3 ferramentas/rig_braco2.py"""
import os,json,struct,io,numpy as np,scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));nrm=np.linalg.norm
def loadglb(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b0=28+l
    def acc(i):
        a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=b0+bv.get('byteOffset',0)+a.get('byteOffset',0)
        ct={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        it=np.dtype(ct).itemsize;st=bv.get('byteStride',it*n)
        return np.ndarray((a['count'],n),ct,buffer=d,offset=off,strides=(st,it)).copy()
    return d,j,b0,acc
def img(d,j,b0,k):
    im=j['images'][j['textures'][k]['source']];bv=j['bufferViews'][im['bufferView']];o=b0+bv.get('byteOffset',0);return Image.open(io.BytesIO(d[o:o+bv['byteLength']]))
def rot_between(a,b_):
    a=a/nrm(a);b_=b_/nrm(b_);v=np.cross(a,b_);c=a@b_
    if nrm(v)<1e-9:return np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);return np.eye(3)+vx+vx@vx*(1/(1+c))
clean=lambda s:s.replace('mixamorig:','').replace('mixamorig','')
# --- esqueleto ---
d,j,b0,acc=loadglb(os.path.join(ROOT,'aethelred_personagem.glb'))
sk=j['skins'][0];names=[clean(j['nodes'][i]['name']) for i in sk['joints']];bi={n:i for i,n in enumerate(names)};nb=len(names)
IB=acc(sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1).astype(np.float64);BW=np.linalg.inv(IB);BP={n:BW[bi[n]][:3,3] for n in names}
# --- braço novo ---
dA,jA,bA,accA=loadglb(os.path.join(ROOT,'modelos','mutated_arm_shoulder_leve.glb'));pA=jA['meshes'][0]['primitives'][0]
X=accA(pA['attributes']['POSITION']).astype(np.float64);NA=accA(pA['attributes']['NORMAL']).astype(np.float64)
UVA=accA(pA['attributes']['TEXCOORD_0']).astype(np.float32);IA=accA(pA['indices']).reshape(-1,3).astype(int)
def cent(z,w=.03,xmin=-9):
    m=(np.abs(X[:,2]-z)<w)&(X[:,0]>xmin);return X[m].mean(0)
Sa=np.array([.19,.08,-.80]);Ea=cent(-.28);Wa=cent(.21)
# --- grafo da malha (vértices soldados) ---
q=np.round(X/1e-5).astype(np.int64);_,inv=np.unique(q,axis=0,return_inverse=True);inv=inv.ravel();nq=inv.max()+1;Tq=inv[IA]
Xq=np.zeros((nq,3));Xq[inv]=X
E=np.vstack([Tq[:,[0,1]],Tq[:,[1,2]],Tq[:,[2,0]]]);EL=nrm(Xq[E[:,0]]-Xq[E[:,1]],axis=1)
G=sp.coo_matrix((np.r_[EL,EL],(np.r_[E[:,0],E[:,1]],np.r_[E[:,1],E[:,0]])),shape=(nq,nq)).tocsr()
Ad=G.copy();Ad.data[:]=1
# --- palma e pontas das garras ---
Ha=np.array([-.17,.05,.40]);palm=np.nonzero(nrm(Xq-Ha,axis=1)<.06)[0]
dp=dijkstra(G,indices=palm,min_only=True);dp[~np.isfinite(dp)]=0
hz=Xq[:,2]>.25;mx=np.where(hz,dp,0.);src=mx.copy()
for _ in range(25):mx=np.maximum(mx,Ad.multiply(mx[None,:]).max(1).toarray().ravel())
cand=np.nonzero(hz&(src>=mx-1e-9)&(src>.35))[0];cand=cand[np.argsort(-src[cand])][:5]
thumb=cand[np.argmax(Xq[cand,0])];rest=[c for c in cand if c!=thumb];rest.sort(key=lambda c:-Xq[c,2])
TIPS={'Index':rest[0],'Middle':rest[1],'Ring':rest[2],'Pinky':rest[3],'Thumb':thumb}
FN=['Pinky','Ring','Middle','Index','Thumb'];D=dijkstra(G,indices=[TIPS[f] for f in FN]);own=np.argmin(D,0)
fpar=np.full(nq,-1.);fown=np.full(nq,-1)
for k,f in enumerate(FN):
    Lf=dp[TIPS[f]]-.2;m=(own==k)&(D[k]<Lf)&hz
    fpar[m]=1-D[k][m]/Lf;fown[m]=k
    print(f,'ponta',Xq[TIPS[f]].round(2),'comprimento',round(Lf,2),'vértices',int(m.sum()))
# --- arquivo -> espaço da malha do personagem (braço direito: +z do arquivo vira -x, polegar +x vira frente +z) ---
Rm=np.array([[0,0,-1],[0,1,0],[1,0,0]],float)
Sb,Eb,Wb=BP['RightArm'],BP['RightForeArm'],BP['RightHand']
s_=(nrm(Eb-Sb)+nrm(Wb-Eb))/(nrm(Ea-Sa)+nrm(Wa-Ea))
toB=lambda P_:(np.asarray(P_)-Sa)@Rm.T*s_+Sb
NJ={'RightArm':toB(Sa),'RightForeArm':toB(Ea),'RightHand':toB(Wa),'RightHandMiddle1':toB(Ha+(Ha-Wa)*.6)}
print('escala',round(s_,5))
# --- proporções do braço antigo: as juntas do braço novo vão para onde estão as do antigo (mesmo comprimento de braço
# e antebraço, mesma direção), e a mão é escalada para alcançar o mesmo ponto que a mão antiga ---
dO,jO_,bO_,accO=loadglb(os.path.join(ROOT,'aethelred_braco_mutante_antigo.glb'))
IBO=np.array(jO_['meshes'][0]['extras']['inverseBind']).reshape(-1,4,4).transpose(0,2,1)
OJ={n:np.linalg.inv(IBO[bi[n]])[:3,3] for n in ('RightArm','RightForeArm','RightHand')}
PO=accO(jO_['meshes'][0]['primitives'][0]['attributes']['POSITION']).astype(float);PO=PO[PO[:,0]<0]
dOw=np.linalg.norm(PO-OJ['RightHand'],axis=1);far=PO[dOw>np.percentile(dOw,99.5)]
OHdir=far.mean(0)-OJ['RightHand'];OHreach=np.percentile(dOw,99.9)
NJ0=dict(NJ)
# --- proporção do braço antigo: antebraço mais grosso, braço um pouco mais grosso, mão e garras maiores ---
# (engrossa em volta do eixo ombro-cotovelo-pulso e aumenta a mão a partir do pulso; ombro, peito e costas ficam iguais)
GROSSO={'braco':1.12,'antebraco':1.45,'mao':0.8,'ombro':1.4}   # mao: multiplica a escala que iguala o alcance da mão antiga
sm0=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
def deform(P_,N_=None):
    P_=np.array(P_,float);No=None if N_ is None else np.array(N_,float);A,Bq,C=NJ0['RightArm'],NJ0['RightForeArm'],NJ0['RightHand']
    oA,oB,oC=OJ['RightArm'],OJ['RightForeArm'],OJ['RightHand']
    out=P_+(oA-A)
    for a,e,oa,oe,k0,k1 in ((A,Bq,oA,oB,GROSSO['braco'],GROSSO['antebraco']),(Bq,C,oB,oC,GROSSO['antebraco'],GROSSO['antebraco'])):
        d=e-a;t=((P_-a)@d)/(d@d);inside=(t>=0)&(t<1)
        k=k0+(k1-k0)*sm0(t)
        if a is A:k=GROSSO['ombro']+(k-GROSSO['ombro'])*sm0(t/.35)
        Rr=rot_between(d,oe-oa);foot=a+np.outer(t,d);perp=(P_-foot)@Rr.T
        out=np.where(inside[:,None],oa+np.outer(t,oe-oa)+perp*k[:,None],out)
        if No is not None:No=np.where(inside[:,None],N_@Rr.T,No)
    d=Bq-A;t=((P_-A)@d)/(d@d);before=t<0;k=1+(GROSSO['ombro']-1)*np.clip(1+t/.35,0,1)
    out=np.where(before[:,None],oA+(P_-A)*k[:,None],out)
    d=C-Bq;t=((P_-Bq)@d)/(d@d);beyond=t>=1
    out=np.where(beyond[:,None],oC+((P_-C)@HR.T)*HS,out)
    if No is not None:No=np.where(beyond[:,None],N_@HR.T,No);return out,No
    return out
# --- pesos ---
Xr=toB(X);Nr=NA@Rm.T
segs=[('RightArm',NJ['RightArm'],NJ['RightForeArm']),('RightForeArm',NJ['RightForeArm'],NJ['RightHand']),('RightHand',NJ['RightHand'],NJ['RightHandMiddle1']),
      ('RightShoulder',BP['RightShoulder'],NJ['RightArm']),('Spine2',BP['Spine2'],BP['Neck'])]
U=nrm(Sb-Wb)
Dd=np.stack([nrm(Xr-(a+np.outer(np.clip(((Xr-a)@(e-a))/max((e-a)@(e-a),1e-20),0,1),e-a)),axis=1) for _,a,e in segs],1)
dmin=Dd.min(1,keepdims=True);Wt=np.where(Dd<dmin*1.35+.01*U,1/np.maximum(Dd,1e-3*U)**4,0);Wt/=Wt.sum(1,keepdims=True)
Wq=np.zeros((nq,len(segs)));cn=np.zeros(nq);np.add.at(Wq,inv,Wt);np.add.at(cn,inv,1);Wq/=cn[:,None]
dg=np.asarray(Ad.sum(1)).ravel()+1
for _ in range(6):Wq=(Ad@Wq+Wq)/dg[:,None]
sm=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
# dedos: mão -> j1 -> j2 -> j3 (juntas extras a partir de nb; direito 0..14, esquerdo 15..29)
nW=len(segs);WF=np.zeros((nq,nW+15));WF[:,:nW]=Wq
for i in np.nonzero(fown>=0)[0]:
    k=fown[i];t=fpar[i];a=sm((t+.05)/.1)
    w1=a*(1-sm((t-.27)/.06));w2=sm((t-.27)/.06)*(1-sm((t-.52)/.06));w3=sm((t-.52)/.06)
    WF[i]=0;WF[i,2]=1-a;WF[i,nW+k*3:nW+k*3+3]=[w1,w2,w3]
bonesR=[bi[n] for n,_,_ in segs]+[nb+i for i in range(15)]
def leftOf(b):
    if b>=nb:return b+15
    n=names[b];return bi['Left'+n[5:]] if n.startswith('Right') else b
Wv=WF[inv];top=np.argsort(-Wv,axis=1)[:,:4];tw=np.take_along_axis(Wv,top,1);tw/=np.maximum(tw.sum(1,keepdims=True),1e-9)
topR=np.array(bonesR)[top];topL=np.vectorize(leftOf)(topR)
# juntas dos dedos no espaço do jogo (centro das fatias em 0, 30%, 55% e na ponta)
dNw=np.linalg.norm(Xr-NJ0['RightHand'],axis=1);hand=((Xr-NJ0['RightForeArm'])@(NJ0['RightHand']-NJ0['RightForeArm']))>((NJ0['RightHand']-NJ0['RightForeArm'])@(NJ0['RightHand']-NJ0['RightForeArm']))
NHdir=Xr[hand&(dNw>np.percentile(dNw[hand],99.5))].mean(0)-NJ0['RightHand'];NHreach=np.percentile(dNw[hand],99.9)
HR=rot_between(NHdir,OHdir);HS=OHreach/NHreach*GROSSO['mao'];print('mão: escala',round(HS,3))
FJ={'Right':{},'Left':{}}
for k,f in enumerate(FN):
    pts=[]
    for fr in (0,.3,.55,1.):
        m=(fown==k)&((np.abs(fpar-fr)<.04) if fr<1 else (fpar>.97));pts.append(deform(toB(Xq[m].mean(0))[None])[0])
    FJ['Right'][f]=[p.tolist() for p in pts];FJ['Left'][f]=[(p*np.array([-1,1,1.])).tolist() for p in pts]
mir=np.array([-1,1,1.]);nv=len(X)
Xr,Nr=deform(Xr,Nr)
NJ={'RightArm':OJ['RightArm'],'RightForeArm':OJ['RightForeArm'],'RightHand':OJ['RightHand'],'RightHandMiddle1':deform(NJ0['RightHandMiddle1'][None])[0]}
IBa=IB.copy()
for n,c in (('RightArm','RightForeArm'),('RightForeArm','RightHand'),('RightHand','RightHandMiddle1')):
    for side,mir in (('Right',np.array([1,1,1.])),('Left',np.array([-1,1,1.]))):
        nn=n.replace('Right',side);cc=c.replace('Right',side)
        do=BP[cc]-BP[nn];dn=(NJ[c]-NJ[n])*mir;M=np.eye(4);M[:3,:3]=rot_between(do,dn)@BW[bi[nn]][:3,:3];M[:3,3]=NJ[n]*mir;IBa[bi[nn]]=np.linalg.inv(M)

Xall=np.vstack([Xr,Xr*mir]).astype(np.float32);Nall=np.vstack([Nr,Nr*mir]).astype(np.float32);UV=np.vstack([UVA,UVA])
Tall=np.vstack([IA,IA[:,[0,2,1]]+nv]).astype(np.uint32)
JF=np.vstack([topR,topL]).astype(np.uint8);WFq=np.clip(np.round(np.vstack([tw,tw])*255),0,255).astype(np.int32);WFq[:,0]+=255-WFq.sum(1);WFq=WFq.astype(np.uint8)
# --- grava (texturas: cor em 4096, normal em 2048 e rugosidade em 1024, JPEG; sem emissão) ---
blobs=[];views=[];accs=[]
def addb(bb):
    off=sum(len(x) for x in blobs);blobs.append(bb+b'\0'*((4-len(bb)%4)%4));views.append({'buffer':0,'byteOffset':off,'byteLength':len(bb)});return len(views)-1
def add(arr,typ,ct,minmax=False,normalized=False):
    a={'bufferView':addb(arr.tobytes()),'componentType':ct,'count':len(arr),'type':typ}
    if normalized:a['normalized']=True
    if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
    accs.append(a);return len(accs)-1
iP=add(Xall,'VEC3',5126,True);iN=add(Nall,'VEC3',5126);iU=add(UV,'VEC2',5126)
iJ=add(JF,'VEC4',5121);iW=add(WFq,'VEC4',5121,normalized=True);iI=add(Tall.ravel(),'SCALAR',5125)
mat=jA['materials'][0];images=[]
COR={'sat':.62,'brilho':1.06}   # cor: menos saturada (vermelho mais rosado/pálido, como o braço antigo) e um pouco mais clara
def addimg(k,size,fmt,cor=False):
    im=img(dA,jA,bA,k).convert('RGB')
    if cor:
        a=np.asarray(im,np.float32);l=(a@np.array([.3,.59,.11],np.float32))[...,None]
        im=Image.fromarray(np.clip((l+(a-l)*COR['sat'])*COR['brilho'],0,255).astype(np.uint8))
    if size:im=im.resize((size,size),Image.LANCZOS)
    bb=io.BytesIO();im.save(bb,fmt,**({'quality':88} if fmt=='JPEG' else {}));images.append({'bufferView':addb(bb.getvalue()),'mimeType':'image/'+fmt.lower()});return len(images)-1
ic=addimg(mat['pbrMetallicRoughness']['baseColorTexture']['index'],None,'JPEG',True)   # cor em 4096: os tendões e cartilagens brancos são detalhes finos
inn=addimg(mat['normalTexture']['index'],2048,'JPEG');imr=addimg(mat['pbrMetallicRoughness']['metallicRoughnessTexture']['index'],1024,'JPEG')
material={'name':'braco','doubleSided':True,'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicRoughnessTexture':{'index':2}},'normalTexture':{'index':1}}
binb=b''.join(blobs)
gl={'asset':{'version':'2.0','generator':'aethelred rig braco 2'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'bracos','mesh':0}],
 'meshes':[{'name':'bracos','primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU,'JOINTS_0':iJ,'WEIGHTS_0':iW},'indices':iI,'material':0}],
   'extras':{'inverseBind':[float(x) for x in IBa.transpose(0,2,1).ravel()],'fingerJoints':FJ,'extraBase':nb,'fingerNames':FN}}],
 'materials':[material],'textures':[{'source':ic},{'source':inn},{'source':imr}],'images':images,'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(binb)}]}
js=json.dumps(gl,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
F=os.path.join(ROOT,'aethelred_braco_mutante_rig.glb')
open(F,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb)
print(F,28+len(js)+len(binb))
