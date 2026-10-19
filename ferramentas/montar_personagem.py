"""Gera aethelred_personagem.glb: o arquivo do Mixamo sem nenhuma alteração (modelos/rogue_mixamo_t_pose.glb, convertido
do FBX com FBX2glTF: malha, esqueleto, pesos e material dele) mais as animações embutidas de aethelred_rig_leve.glb (Idle, Walk, Sprint, Jump Start, Jump, Jump Land) passadas para
este esqueleto: rotações copiadas osso a osso pelo nome (os dois são esqueletos do Mixamo); o quadril copia a
orientação no mundo e o deslocamento dele é escalado pela altura do quadril; posições dos outros ossos e escalas
saem (eram as proporções do modelo antigo). Também grava em extras.armHide os vértices dos braços (para a troca
pelos braços mutantes). Uso: python3 ferramentas/montar_personagem.py"""
import os,json,struct,base64,numpy as np
from scipy.spatial.transform import Rotation as Rr
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def readglb(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);bl=struct.unpack('<I',d[20+l:24+l])[0];return j,bytearray(d[28+l:28+l+bl])
def acc(j,b,i):
    a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    ct={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    es=np.dtype(ct).itemsize*n;st=bv.get('byteStride',es)
    if st==es:return np.frombuffer(bytes(b),dtype=ct,count=a['count']*n,offset=off).reshape(a['count'],n)
    raw=np.frombuffer(bytes(b),dtype=np.uint8,count=st*(a['count']-1)+es,offset=off)
    return np.lib.stride_tricks.as_strided(raw,shape=(a['count'],es),strides=(st,1)).copy().view(ct).reshape(a['count'],n)
clean=lambda s:s.replace('mixamorig:','').replace('mixamorig','')
def nodeM(n):
    M=np.eye(4)
    if 'matrix' in n:return np.array(n['matrix']).reshape(4,4).T
    S=np.diag(list(n.get('scale',[1,1,1]))+[1]);R=np.eye(4)
    if 'rotation' in n:R[:3,:3]=Rr.from_quat(n['rotation']).as_matrix()
    T=np.eye(4);T[:3,3]=n.get('translation',[0,0,0]);return T@R@S
def worldM(j,k):
    par={c:i for i,n in enumerate(j['nodes']) for c in n.get('children',[])};M=np.eye(4)
    while k is not None:M=nodeM(j['nodes'][k])@M;k=par.get(k)
    return M
jN,bN=readglb(os.path.join(ROOT,'modelos','rogue_mixamo_t_pose.glb'))
jO,bO=readglb(os.path.join(ROOT,'aethelred_rig_leve.glb'))
blobs=[bytes(bN)]
def addview(data,target=None):
    off=sum(len(x) for x in blobs);data=bytes(data);pad=(4-len(data)%4)%4;blobs.append(data+b'\0'*pad)
    v={'buffer':0,'byteOffset':off,'byteLength':len(data)}
    if target:v['target']=target
    jN['bufferViews'].append(v);return len(jN['bufferViews'])-1
def addacc(arr,typ,ct,minmax=False):
    bv=addview(arr.tobytes());a={'bufferView':bv,'componentType':ct,'count':len(arr),'type':typ}
    if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
    jN['accessors'].append(a);return len(jN['accessors'])-1
# malha, esqueleto, pesos e material ficam exatamente como saíram do Mixamo (só as animações são acrescentadas)
# --- animações antigas no esqueleto novo ---
nodeN={clean(n.get('name','')):i for i,n in enumerate(jN['nodes'])}
hipN=nodeN['Hips'];hipO=[i for i,n in enumerate(jO['nodes']) if clean(n.get('name',''))=='Hips'][0]
parO=[i for i,n in enumerate(jO['nodes']) if hipO in n.get('children',[])][0]
parN=[i for i,n in enumerate(jN['nodes']) if hipN in n.get('children',[])][0]
MpO=worldM(jO,parO);MpN=worldM(jN,parN)
def rotpart(M):
    R=M[:3,:3].copy();return R/np.cbrt(np.linalg.det(R))
RpO=rotpart(MpO);RpN=rotpart(MpN)
sO=np.cbrt(np.linalg.det(MpO[:3,:3]));sN=np.cbrt(np.linalg.det(MpN[:3,:3]))
# altura do quadril no mundo (acima dos pés) nos dois esqueletos, para escalar o deslocamento
def jointW(j,name):
    k=[i for i,n in enumerate(j['nodes']) if clean(n.get('name',''))==name][0];return worldM(j,k)[:3,3]
hO=jointW(jO,'Hips')[1]-min(jointW(jO,'LeftToeBase')[1],jointW(jO,'LeftFoot')[1])
hN=jointW(jN,'Hips')[1]-min(jointW(jN,'LeftToeBase')[1],jointW(jN,'LeftFoot')[1])
restO=MpO@np.append(np.array(jO['nodes'][hipO].get('translation',[0,0,0])),1)
restN=np.array(jN['nodes'][hipN].get('translation',[0,0,0]))
print('altura do quadril (mundo): antigo',round(hO,4),'novo',round(hN,4))
# retarget: R_novo(t)=R_velho(t)*R0v^-1*Swing*R0n (no mundo). Nos braços Swing alinha o eixo do osso antigo ao novo
# (os dois repousos diferem: A e T); no resto Swing=identidade (o osso gira a partir do repouso como o antigo girava)
def slerp_sample(T,Q,t):
    k=np.clip(np.searchsorted(T,t)-1,0,len(T)-2);u=np.clip((t-T[k])/np.maximum(T[k+1]-T[k],1e-9),0,1)
    q0=Q[k];q1=Q[k+1].copy();neg=(q0*q1).sum(1)<0;q1[neg]*=-1;q=q0*(1-u)[:,None]+q1*u[:,None];return q/nrm(q,axis=1)[:,None]
def lerp_sample(T,V,t):
    k=np.clip(np.searchsorted(T,t)-1,0,len(T)-2);u=np.clip((t-T[k])/np.maximum(T[k+1]-T[k],1e-9),0,1);return V[k]*(1-u)[:,None]+V[k+1]*u[:,None]
nrm=np.linalg.norm
def chainNames(j,root):
    out=[];kids={}
    def rec(k,p):
        nm=clean(j['nodes'][k].get('name',''));out.append((nm,k,p))
        for c in j['nodes'][k].get('children',[]):rec(c,k)
    rec(root,None);return out
ordO=chainNames(jO,hipO);ordN=chainNames(jN,hipN)
nodeO={nm:k for nm,k,_ in ordO}
def restWorldR(j,order,Rpar):
    W={}
    for nm,k,p in order:
        Rl=Rr.from_quat(j['nodes'][k].get('rotation',[0,0,0,1])).as_matrix()
        W[nm]=(W[clean(j['nodes'][p]['name'])] if p is not None else Rpar)@Rl
    return W
R0v=restWorldR(jO,ordO,RpO);R0n=restWorldR(jN,ordN,RpN)
def yaxis(R):return R[:,1]
OFF={}
for nm,_,_ in ordN:
    if nm not in R0v:continue
    sw=np.eye(3)
    a_=R0n[nm][:,1];b_=R0v[nm][:,1]
    v=np.cross(a_,b_);c=a_@b_
    if nrm(v)>1e-8:vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);sw=np.eye(3)+vx+vx@vx/(1+c)
    # braços (repouso em A no esqueleto antigo e em T no novo): pela direção; o resto (tronco, cabeça, pernas): pela
    # rotação a partir do repouso, porque a cabeça do esqueleto antigo já ficava inclinada para trás no repouso
    armlike=any(x in nm for x in ['Arm','Hand'])
    OFF[nm]=R0v[nm].T@(sw if armlike else np.eye(3))@R0n[nm]
jN['animations']=[]
for an in jO['animations']:
    rot={};tr=None;times=set()
    for ch in an['channels']:
        nm=clean(jO['nodes'][ch['target']['node']].get('name',''));sm=an['samplers'][ch['sampler']]
        T=acc(jO,bO,sm['input']).ravel().astype(np.float64);V=acc(jO,bO,sm['output']).astype(np.float64)
        if ch['target']['path']=='rotation':rot[nm]=(T,V);times.update(T.tolist())
        elif ch['target']['path']=='translation' and nm=='Hips':tr=(T,V)
    t=np.array(sorted(times));F=len(t)
    WO={}
    for nm,k,p in ordO:
        if nm in rot:Rl=Rr.from_quat(slerp_sample(rot[nm][0],rot[nm][1],t)).as_matrix()
        else:Rl=np.repeat(Rr.from_quat(jO['nodes'][k].get('rotation',[0,0,0,1])).as_matrix()[None],F,0)
        Pw=WO[clean(jO['nodes'][p]['name'])] if p is not None else np.repeat(RpO[None],F,0)
        WO[nm]=Pw@Rl
    WN={};chans=[];samps=[]
    iT=addacc(t.astype(np.float32),'SCALAR',5126,True)
    for nm,k,p in ordN:
        Pw=WN[clean(jN['nodes'][p]['name'])] if p is not None else np.repeat(RpN[None],F,0)
        if nm in WO and nm in OFF:Wn=WO[nm]@OFF[nm]
        else:Wn=Pw@Rr.from_quat(jN['nodes'][k].get('rotation',[0,0,0,1])).as_matrix()
        WN[nm]=Wn
        if nm not in WO:continue
        q=Rr.from_matrix(np.einsum('nji,njk->nik',Pw,Wn)).as_quat()
        iV=addacc(q.astype(np.float32),'VEC4',5126);samps.append({'input':iT,'output':iV,'interpolation':'LINEAR'})
        chans.append({'sampler':len(samps)-1,'target':{'node':nodeN[nm],'path':'rotation'}})
    if tr:
        T,V=tr;Wp=(MpO@np.hstack([V,np.ones((len(V),1))]).T).T[:,:3];dW=(Wp-restO[:3])*(hN/hO);Vn=restN+(RpN.T@dW.T).T/sN
        iT2=addacc(T.astype(np.float32),'SCALAR',5126,True);iV=addacc(Vn.astype(np.float32),'VEC3',5126)
        samps.append({'input':iT2,'output':iV,'interpolation':'LINEAR'});chans.append({'sampler':len(samps)-1,'target':{'node':hipN,'path':'translation'}})
    jN['animations'].append({'name':an.get('name'),'channels':chans,'samplers':samps})
    print('animação',an.get('name'),len(chans),'canais',F,'quadros')
# --- vértices dos braços (para a troca pelos braços mutantes) ---
mesh=jN['meshes'][0];p=mesh['primitives'][0]
Jv=acc(jN,bN,p['attributes']['JOINTS_0']).astype(int);Wv=acc(jN,bN,p['attributes']['WEIGHTS_0']).astype(np.float64)
if Wv.max()>1.5:Wv=Wv/255.
skin=jN['skins'][0];jn=[clean(jN['nodes'][i]['name']) for i in skin['joints']]
armIdx=[i for i,n in enumerate(jn) if any(s in n for s in ['Arm','Hand']) and 'Shoulder' not in n]
aw=(np.isin(Jv,armIdx)*Wv).sum(1)
mesh['extras']={'armHide':base64.b64encode(np.packbits((aw>.5).astype(np.uint8),bitorder='little').tobytes()).decode()}
print('vértices de braço',int((aw>.5).sum()),'de',len(aw))
# --- grava ---
binb=b''.join(blobs);jN['buffers']=[{'byteLength':len(binb)}]
js=json.dumps(jN,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb
F=os.path.join(ROOT,'aethelred_personagem.glb');open(F,'wb').write(out);print(F,len(out))
