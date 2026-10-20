"""Acrescenta ao aethelred_personagem.glb uma animação do Mixamo baixada para este personagem (mesmo esqueleto).
Uso: python3 ferramentas/anim_mixamo.py arquivo.fbx NomeDaAnimacao [--renomear Antigo:Novo]
--renomear troca o nome de uma animação que já está no arquivo (ex.: Idle:Guarda) antes de acrescentar a nova.
Os canais são copiados osso a osso pelo nome; a translação do quadril é ajustada pela altura de repouso."""
import os,sys,json,struct,subprocess,tempfile,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FBX2GLTF=os.environ.get('FBX2GLTF','FBX2glTF')
args=sys.argv[1:];ren=None
if '--renomear' in args:i=args.index('--renomear');ren=args[i+1].split(':');del args[i:i+2]
fbx,name=args
def rd(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);bl=struct.unpack('<I',d[20+l:24+l])[0];return j,bytearray(d[28+l:28+l+bl])
def acc(j,b,i):
    a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    ct={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    return np.frombuffer(bytes(b[off:off+a['count']*n*np.dtype(ct).itemsize]),dtype=ct).reshape(a['count'],n)
F=os.path.join(ROOT,'aethelred_personagem.glb');j,b=rd(F)
if ren:
    for a in j['animations']:
        if a.get('name')==ren[0]:a['name']=ren[1];print('renomeada',ren[0],'->',ren[1])
assert not any(a.get('name')==name for a in j['animations']),'já existe uma animação '+name
def addacc(arr,typ,minmax=False):
    while len(b)%4:b.append(0)
    j['bufferViews'].append({'buffer':0,'byteOffset':len(b),'byteLength':arr.nbytes});b.extend(arr.tobytes())
    a={'bufferView':len(j['bufferViews'])-1,'componentType':5126,'count':len(arr),'type':typ}
    if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
    j['accessors'].append(a);return len(j['accessors'])-1
tmp=tempfile.mkdtemp();out=os.path.join(tmp,'a')
subprocess.run([FBX2GLTF,'-b','-i',fbx,'-o',out],check=True,capture_output=True)
jA,bA=rd(out+'.glb');an=jA['animations'][0]
nodeN={n.get('name',''):i for i,n in enumerate(j['nodes'])}
hN=j['nodes'][nodeN['mixamorig:Hips']].get('translation',[0,0,0])[1]
hA=[n for n in jA['nodes'] if n.get('name')=='mixamorig:Hips'][0].get('translation',[0,1,0])[1];k=hN/hA
chans=[];samps=[]
for ch in an['channels']:
    nm=jA['nodes'][ch['target']['node']].get('name','');path=ch['target']['path']
    if nm not in nodeN or path=='scale' or (path=='translation' and nm!='mixamorig:Hips'):continue
    sm=an['samplers'][ch['sampler']];T=acc(jA,bA,sm['input']).ravel().astype(np.float32).copy();V=acc(jA,bA,sm['output']).astype(np.float32).copy()
    if path=='translation':V*=k
    samps.append({'input':addacc(T.reshape(-1,1),'SCALAR',True),'output':addacc(V,'VEC4' if path=='rotation' else 'VEC3'),'interpolation':sm.get('interpolation','LINEAR')})
    chans.append({'sampler':len(samps)-1,'target':{'node':nodeN[nm],'path':path}})
j['animations'].append({'name':name,'channels':chans,'samplers':samps})
while len(b)%4:b.append(0)
j['buffers'][0]['byteLength']=len(b)
js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
open(F,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(b))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(b),0x004E4942)+bytes(b))
print('animação',name,len(chans),'canais; animações:',[a['name'] for a in j['animations']])
