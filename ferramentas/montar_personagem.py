"""Gera aethelred_personagem.glb: o arquivo do Mixamo sem nenhuma alteração (modelos/rogue_mixamo_t_pose.glb, convertido
do FBX com FBX2glTF: malha, esqueleto, pesos e material dele) mais as animações básicas do Mixamo baixadas com este personagem (animacoes/base: idle, walking, sprint, jumping_up). Também grava em extras.armHide os vértices dos braços (para a troca
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
# --- animações básicas do Mixamo (animacoes/base/*.fbx, baixadas com este personagem, convertidas com FBX2glTF) ---
# mesmo esqueleto: os canais são copiados osso a osso pelo nome; a translação do quadril é ajustada pela razão entre a
# altura de repouso do quadril deste arquivo e a do arquivo da animação
import subprocess,tempfile
FBX2GLTF=os.environ.get('FBX2GLTF','FBX2glTF')
BASE=[('idle.fbx','Idle'),('walking.fbx','Walk'),('sprint.fbx','Sprint'),('jumping_up.fbx','Jump')]
nodeN={n.get('name',''):i for i,n in enumerate(jN['nodes'])}
hipN=nodeN['mixamorig:Hips'];hN=jN['nodes'][hipN].get('translation',[0,0,0])[1]
jN['animations']=[]
tmp=tempfile.mkdtemp()
for fn,name in BASE:
    out=os.path.join(tmp,name);subprocess.run([FBX2GLTF,'-b','-i',os.path.join(ROOT,'animacoes','base',fn),'-o',out],check=True,capture_output=True)
    jA,bA=readglb(out+'.glb');an=jA['animations'][0]
    hipA=[n for n in jA['nodes'] if n.get('name')=='mixamorig:Hips'][0];k=hN/hipA.get('translation',[0,1,0])[1]
    chans=[];samps=[]
    for ch in an['channels']:
        nm=jA['nodes'][ch['target']['node']].get('name','');path=ch['target']['path']
        if nm not in nodeN or path=='scale' or (path=='translation' and nm!='mixamorig:Hips'):continue
        sm=an['samplers'][ch['sampler']];T=acc(jA,bA,sm['input']).ravel().astype(np.float32);V=acc(jA,bA,sm['output']).astype(np.float32)
        if path=='translation':
            V=V*k
            # sem deslocamento: tira o avanço linear do quadril (x e z) ao longo do ciclo; quem move o personagem é o jogo
            if name in ('Walk','Sprint'):
                u=((T-T[0])/max(T[-1]-T[0],1e-6))[:,None];V[:,[0,2]]-=V[0,[0,2]]+(V[-1,[0,2]]-V[0,[0,2]])*u
        iT=addacc(T,'SCALAR',5126,True);iV=addacc(V,'VEC4' if path=='rotation' else 'VEC3',5126)
        samps.append({'input':iT,'output':iV,'interpolation':sm.get('interpolation','LINEAR')})
        chans.append({'sampler':len(samps)-1,'target':{'node':nodeN[nm],'path':path}})
    jN['animations'].append({'name':name,'channels':chans,'samplers':samps})
    print('animação',name,'de',fn,len(chans),'canais')
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
