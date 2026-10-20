"""Dedos dos braços mutantes (aethelred_braco_mutante_rig.glb): cada dedo e sua garra ganham 3 juntas próprias,
no caminho do próprio dedo (da base até a ponta da garra), em vez de usar os ossos dos dedos humanos do esqueleto
(curtos, e que puxavam os pedaços das garras para cima uns dos outros).
1) dedos separados pelo ângulo em leque a partir da palma (mindinho, anelar, médio, indicador) e o polegar à parte;
2) em cada dedo, distância ao longo da malha a partir da base; juntas em 0, 30% e 55% do caminho;
3) pesos: cada vértice só nas juntas do seu dedo (transição suave entre elas e da mão para a 1ª junta).
Os ossos extras são criados no jogo (filhos do osso da mão); o arquivo guarda as posições em extras.fingerJoints
e os índices JOINTS a partir de extras.extraBase. Uso: python3 ferramentas/dedos_mutantes.py (uma vez)"""
import os,json,struct,numpy as np,scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def rd(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);return j,bytearray(d[28+l:])
def view(j,b,i):
    a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];ct={5126:np.float32,5121:np.uint8,5123:np.uint16,5125:np.uint32}[a['componentType']]
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];o=v.get('byteOffset',0)+a.get('byteOffset',0)
    it=np.dtype(ct).itemsize;st=v.get('byteStride',it*n)                 # dados intercalados: respeita o passo
    return np.ndarray((a['count'],n),ct,buffer=b,offset=o,strides=(st,it))
jS,bS=rd(os.path.join(ROOT,'modelos','mutated_arm_claw_leve.glb'))
X=view(jS,bS,jS['meshes'][0]['primitives'][0]['attributes']['POSITION']).astype(float)
IA=view(jS,bS,jS['meshes'][0]['primitives'][0]['indices']).reshape(-1,3).astype(int)
F=os.path.join(ROOT,'aethelred_braco_mutante_rig.glb');j,b=rd(F)
pr=j['meshes'][0]['primitives'][0];P=view(j,b,pr['attributes']['POSITION']).astype(float)
J=view(j,b,pr['attributes']['JOINTS_0']);W=view(j,b,pr['attributes']['WEIGHTS_0'])
nv=len(X);assert len(P)==2*nv
jB,_=rd(os.path.join(ROOT,'aethelred_personagem.glb'))
names=[jB['nodes'][i]['name'].replace('mixamorig:','') for i in jB['skins'][0]['joints']];nb=len(names)
HAND={'Right':names.index('RightHand'),'Left':names.index('LeftHand')}
# --- segmentação no arquivo de origem (mão em +z, palma para baixo, polegar em +x) ---
Ha=X[np.abs(X[:,2]-.42)<.06].mean(0);dH=np.linalg.norm(X-Ha,axis=1)
thumb=(dH>.18)&(X[:,0]>.3)&(X[:,2]<.66)&(X[:,1]<.45)
th=np.degrees(np.arctan2(X[:,0]-.1,X[:,2]-.4))
ZK=.6   # nós dos dedos
fing=(X[:,2]>ZK-.04)&~thumb
cuts=[-24,-3,18]                                     # vales entre mindinho | anelar | médio | indicador
fid=np.digitize(th,cuts)                             # 0 mindinho, 1 anelar, 2 médio, 3 indicador
FN=['Pinky','Ring','Middle','Index','Thumb']
groups={FN[k]:np.nonzero(fing&(fid==k))[0] for k in range(4)};groups['Thumb']=np.nonzero(thumb)[0]
# grafo da malha (vértices soldados)
qk=np.round(X/1e-5).astype(np.int64);_,inv=np.unique(qk,axis=0,return_inverse=True);inv=inv.ravel();nq=inv.max()+1
T=inv[IA];E=np.vstack([T[:,[0,1]],T[:,[1,2]],T[:,[2,0]]]);
Xq=np.zeros((nq,3));Xq[inv]=X
L=np.linalg.norm(Xq[E[:,0]]-Xq[E[:,1]],axis=1)
joints={};tparam=np.full(nv,-1.0);owner=np.full(nv,-1)
for k,f in enumerate(FN):
    idx=groups[f];qs=np.unique(inv[idx]);mk=np.zeros(nq,bool);mk[qs]=True
    e=E[mk[E].all(1)];l=np.linalg.norm(Xq[e[:,0]]-Xq[e[:,1]],axis=1)
    G=sp.coo_matrix((np.r_[l,l],(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(nq,nq)).tocsr()
    base=qs[(Xq[qs,2]<ZK+.03)] if f!='Thumb' else qs[np.argsort(np.linalg.norm(Xq[qs]-Ha,axis=1))[:40]]
    D=dijkstra(G,indices=base,min_only=True)
    dq=D[qs];ok=np.isfinite(dq)
    if (~ok).any():   # pedaços soltos: distância do vizinho alcançável mais próximo
        from scipy.spatial import cKDTree
        tr=cKDTree(Xq[qs[ok]]);_,nn=tr.query(Xq[qs[~ok]]);dq[~ok]=dq[ok][nn]
    dmax=np.percentile(dq,99.5);t=np.clip(dq/dmax,0,1);tq=np.zeros(nq);tq[qs]=t
    tv=tq[inv[idx]];tparam[idx]=tv;owner[idx]=k
    pts=[]
    for fr in (0,.3,.55,1.0):
        sel=np.abs(tv-fr)<(.04 if fr<1 else .02)
        if fr==1.0:sel=tv>.97
        pts.append(idx[np.nonzero(sel)[0]])
    joints[f]=pts
    print(f,'vértices',len(idx),'comprimento',round(dmax,3))
# --- pesos novos (lado direito = primeiros nv vértices; o esquerdo é espelho) ---
sm=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
Jn=J.copy();Wn=W.astype(float)/255
for half,side in ((0,'Right'),(1,'Left')):
    off=half*nv;Wh=Wn[off:off+nv];Jh=Jn[off:off+nv]
    # tira pesos de ossos de dedos humanos: vão para a mão
    for c in range(4):
        isF=np.array(['Hand' in names[x] and names[x]!=side+'Hand' and names[x]!=('Left' if side=='Right' else 'Right')+'Hand' for x in Jh[:,c]])
        Jh[isF,c]=HAND[side]
    for i in np.nonzero(owner>=0)[0]:
        k=owner[i];t=tparam[i];base=nb+(0 if side=='Right' else 15)+k*3
        # pesos ao longo do dedo: mão -> j1 -> j2 -> j3
        w=np.zeros(4);bones=[HAND[side],base,base+1,base+2]
        a=sm((t+.05)/.1)                                  # entra no dedo logo depois da base
        w1=a*(1-sm((t-.27)/.06));w2=sm((t-.27)/.06)*(1-sm((t-.52)/.06));w3=sm((t-.52)/.06)
        w=np.array([1-a,w1,w2,w3]);w/=w.sum()
        Jh[i]=bones;Wh[i]=w
    Wn[off:off+nv]=Wh;Jn[off:off+nv]=Jh
Wq=np.round(Wn*255).astype(int);Wq[:,0]+=255-Wq.sum(1);W[:]=np.clip(Wq,0,255).astype(np.uint8);J[:]=Jn
# posições das juntas no espaço do arquivo do jogo (centro de cada fatia)
FJ={}
for side,off,mir in (('Right',0,1),('Left',nv,-1)):
    FJ[side]={f:[P[off+np.asarray(idx)].mean(0).tolist() for idx in joints[f]] for f in FN}
ex=j['meshes'][0].setdefault('extras',{});ex['fingerJoints']=FJ;ex['extraBase']=nb;ex['fingerNames']=FN
js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
open(F,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(b))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(b),0x004E4942)+bytes(b))
print('ok',F)
