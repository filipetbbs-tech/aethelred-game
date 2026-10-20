"""Braço mutante misto: o braço e a mão do modelo antigo (aethelred_braco_mutante_antigo.glb) com o ombro, o peito e as
costas do modelo novo (aethelred_braco_mutante_rig.glb), que cobrem a junção com o tronco. Os dois pedaços ficam no
mesmo arquivo (uma malha com duas partes, cada uma com a sua textura) e usam o mesmo esqueleto e as matrizes de ligação
do antigo; os vértices do ombro novo são convertidos para essas matrizes mantendo a mesma posição em repouso.
CORTE: até onde (fração do ombro ao cotovelo) vai o pedaço do ombro novo. Uso: python3 ferramentas/braco_misto.py"""
import os,json,struct,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORTE=.28
def rd(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];return json.loads(d[20:20+l]),d[28+l:]
def acc(j,b,i):
    a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];ct={5126:np.float32,5121:np.uint8,5123:np.uint16,5125:np.uint32}[a['componentType']]
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']];o=v.get('byteOffset',0)+a.get('byteOffset',0);it=np.dtype(ct).itemsize
    return np.ndarray((a['count'],n),ct,buffer=b,offset=o,strides=(v.get('byteStride',it*n),it)).copy()
jO,bO=rd(os.path.join(ROOT,'aethelred_braco_mutante_antigo.glb'));jN,bN=rd(os.path.join(ROOT,'aethelred_braco_mutante_rig.glb'))
jB,_=rd(os.path.join(ROOT,'aethelred_personagem.glb'))
jsk=jB['skins'][0];names=[jB['nodes'][i]['name'].replace('mixamorig:','') for i in jsk['joints']];bi={n:i for i,n in enumerate(names)}
dB=open(os.path.join(ROOT,'aethelred_personagem.glb'),'rb').read();lB=struct.unpack('<I',dB[12:16])[0]
IBb=acc(jB,dB[28+lB:],jsk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1).astype(float);BW=np.linalg.inv(IBb)
exO=jO['meshes'][0]['extras'];exN=jN['meshes'][0]['extras']
IBO=np.array(exO['inverseBind']).reshape(-1,4,4).transpose(0,2,1);IBN=np.array(exN['inverseBind']).reshape(-1,4,4).transpose(0,2,1)
pN=jN['meshes'][0]['primitives'][0];aN=pN['attributes']
P=acc(jN,bN,aN['POSITION']).astype(float);Nn=acc(jN,bN,aN['NORMAL']).astype(float);UV=acc(jN,bN,aN['TEXCOORD_0']);J=acc(jN,bN,aN['JOINTS_0']).astype(int);W=acc(jN,bN,aN['WEIGHTS_0']).astype(float)/255
T=acc(jN,bN,pN['indices']).reshape(-1,3)
# pedaço do ombro: em cada lado, vértices antes de CORTE do caminho ombro->cotovelo (pela junta do modelo novo)
keep=np.zeros(len(P),bool)
for s in ('Right','Left'):
    A=np.linalg.inv(IBN[bi[s+'Arm']])[:3,3];E=np.linalg.inv(IBN[bi[s+'ForeArm']])[:3,3];d=E-A
    side=(P[:,0]<0) if s=='Right' else (P[:,0]>0)
    t=((P-A)@d)/(d@d);keep|=side&(t<CORTE)
keep&=(J<len(IBO)).all(1)|(W<=0).all(1)   # só ossos do esqueleto (os dos dedos não entram no ombro)
keep&=np.array([all(w<=0 or k<len(IBO) for k,w in zip(jj,ww)) for jj,ww in zip(J,W)])
tk=keep[T].all(1);T2=T[tk];used=np.unique(T2);remap=-np.ones(len(P),int);remap[used]=np.arange(len(used))
P2,N2,UV2,J2,W2=P[used],Nn[used],UV[used],J[used],W[used];T2=remap[T2]
# converte para as matrizes de ligação do antigo, mantendo a posição em repouso
def M(IB,jj,ww):return sum(w*(BW[k] if k<len(BW) else np.eye(4))@IB[k] for k,w in zip(jj,ww) if w>0)
for i in range(len(P2)):
    D=M(IBN,J2[i],W2[i])@np.r_[P2[i],1];Mo=M(IBO,J2[i],W2[i]);P2[i]=np.linalg.solve(Mo,D)[:3]
    n=np.linalg.inv(Mo[:3,:3])@(M(IBN,J2[i],W2[i])[:3,:3]@N2[i]);N2[i]=n/np.linalg.norm(n)
print('ombro novo: vértices',len(P2),'triângulos',len(T2))
# --- grava: primitiva 0 = braço antigo inteiro, primitiva 1 = ombro novo ---
blobs=[];views=[];accs=[]
def addb(bb):
    off=sum(len(x) for x in blobs);blobs.append(bb+b'\0'*((4-len(bb)%4)%4));views.append({'buffer':0,'byteOffset':off,'byteLength':len(bb)});return len(views)-1
def add(arr,typ,ct,minmax=False,normalized=False,stride=None):
    a={'bufferView':addb(np.ascontiguousarray(arr).tobytes()),'componentType':ct,'count':len(arr),'type':typ}
    if stride:views[a['bufferView']]['byteStride']=stride
    if normalized:a['normalized']=True
    if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
    accs.append(a);return len(accs)-1
# normais em int8 e UV em uint16 (KHR_mesh_quantization): o arquivo precisa caber no link de teste
def qn(n):n=np.asarray(n,float);n=n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);o=np.zeros((len(n),4),np.int8);o[:,:3]=np.round(n*127);return o
def quv(u):return np.round(np.clip(np.asarray(u,float),0,1)*65535).astype(np.uint16)
def prim(j,b,p,mat):
    a=p['attributes']
    return {'attributes':{'POSITION':add(acc(j,b,a['POSITION']).astype(np.float32),'VEC3',5126,True),'NORMAL':add(qn(acc(j,b,a['NORMAL'])),'VEC3',5120,normalized=True,stride=4),
      'TEXCOORD_0':add(quv(acc(j,b,a['TEXCOORD_0'])),'VEC2',5123,normalized=True),'JOINTS_0':add(acc(j,b,a['JOINTS_0']).astype(np.uint8),'VEC4',5121),
      'WEIGHTS_0':add(acc(j,b,a['WEIGHTS_0']).astype(np.uint8),'VEC4',5121,normalized=True)},'indices':add(acc(j,b,p['indices']).astype(np.uint32).ravel(),'SCALAR',5125),'material':mat}
images=[];textures=[];materials=[]
def copymat(j,b,m):
    m=json.loads(json.dumps(m))
    def tex(ti):
        im=j['images'][j['textures'][ti]['source']];v=j['bufferViews'][im['bufferView']];o=v.get('byteOffset',0)
        data=bytes(b[o:o+v['byteLength']]);mt=im.get('mimeType','image/jpeg')
        if len(data)>600000:   # textura grande: reduz para 2048 (o arquivo precisa caber no link de teste)
            from PIL import Image;import io as _io
            pi=Image.open(_io.BytesIO(data)).convert('RGB');pi=pi.resize((min(2048,pi.width),min(2048,pi.height)),Image.LANCZOS);bb=_io.BytesIO();pi.save(bb,"JPEG",quality=84);data=bb.getvalue();mt='image/jpeg'
        images.append({'bufferView':addb(data),'mimeType':mt});textures.append({'source':len(images)-1});return len(textures)-1
    pb=m.get('pbrMetallicRoughness',{})
    for k in ('baseColorTexture','metallicRoughnessTexture'):
        if k in pb:pb[k]={'index':tex(pb[k]['index'])}
    for k in ('normalTexture','emissiveTexture','occlusionTexture'):
        if k in m:m[k]={'index':tex(m[k]['index'])}
    materials.append(m);return len(materials)-1
p0=prim(jO,bO,jO['meshes'][0]['primitives'][0],copymat(jO,bO,jO['materials'][jO['meshes'][0]['primitives'][0].get('material',0)]))
m1=copymat(jN,bN,jN['materials'][pN.get('material',0)])
Wq=np.clip(np.round(W2*255),0,255).astype(np.int32);Wq[:,0]+=255-Wq.sum(1)
p1={'attributes':{'POSITION':add(P2.astype(np.float32),'VEC3',5126,True),'NORMAL':add(qn(N2),'VEC3',5120,normalized=True,stride=4),'TEXCOORD_0':add(quv(UV2),'VEC2',5123,normalized=True),
   'JOINTS_0':add(J2.astype(np.uint8),'VEC4',5121),'WEIGHTS_0':add(Wq.astype(np.uint8),'VEC4',5121,normalized=True)},'indices':add(T2.astype(np.uint32).ravel(),'SCALAR',5125),'material':m1}
binb=b''.join(blobs)
gl={'asset':{'version':'2.0','generator':'aethelred braco misto'},'extensionsUsed':['KHR_mesh_quantization'],'extensionsRequired':['KHR_mesh_quantization'],'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'bracos','mesh':0}],
 'meshes':[{'name':'bracos','primitives':[p0,p1],'extras':exO}],'materials':materials,'textures':textures,'images':images,'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(binb)}]}
js=json.dumps(gl,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
F=os.path.join(ROOT,'aethelred_braco_mutante_misto.glb')
open(F,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb);print(F,28+len(js)+len(binb))
