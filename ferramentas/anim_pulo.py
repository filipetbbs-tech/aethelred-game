"""Embute a animação de pulo completa do Mixamo (animacoes/base/jumping.fbx: agacha, sai do chão, voa, pousa e se
recupera) em aethelred_personagem.glb como "Jumping". O quadril não sobe acima da altura de pé (a altura no ar vem do
jogo; da animação ficam só os agachamentos) e não anda para os lados.
Uso: FBX2GLTF=/caminho/FBX2glTF python3 ferramentas/anim_pulo.py"""
import os,json,struct,subprocess,tempfile,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));F=os.path.join(ROOT,'aethelred_personagem.glb')
def rd(f):
    d=open(f,'rb').read();l=struct.unpack('<I',d[12:16])[0];return json.loads(d[20:20+l]),bytearray(d[28+l:])
def acc(j,b,i):
    a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
    return np.frombuffer(bytes(b),np.float32,a['count']*n,v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,n).copy()
j,b=rd(F)
def addacc(arr,typ,minmax=False):
    while len(b)%4:b.append(0)
    o=len(b);b.extend(arr.astype(np.float32).tobytes())
    j['bufferViews'].append({'buffer':0,'byteOffset':o,'byteLength':arr.nbytes})
    a={'bufferView':len(j['bufferViews'])-1,'componentType':5126,'count':len(arr),'type':typ}
    if minmax:a['min']=[float(arr.min())];a['max']=[float(arr.max())]
    j['accessors'].append(a);return len(j['accessors'])-1
nodeN={n.get('name',''):i for i,n in enumerate(j['nodes'])}
hN=j['nodes'][nodeN['mixamorig:Hips']].get('translation',[0,0,0])[1]
tmp=tempfile.mkdtemp();out=os.path.join(tmp,'j')
subprocess.run([os.environ.get('FBX2GLTF','FBX2glTF'),'-b','-i',os.path.join(ROOT,'animacoes','base','jumping.fbx'),'-o',out],check=True,capture_output=True)
jA,bA=rd(out+'.glb');an=jA['animations'][0]
hipA=[n for n in jA['nodes'] if n.get('name')=='mixamorig:Hips'][0];k=hN/hipA.get('translation',[0,1,0])[1]
chans=[];samps=[]
for ch in an['channels']:
    nm=jA['nodes'][ch['target']['node']].get('name','');path=ch['target']['path']
    if nm not in nodeN or path=='scale' or (path=='translation' and nm!='mixamorig:Hips'):continue
    sm=an['samplers'][ch['sampler']];T=acc(jA,bA,sm['input']).ravel();V=acc(jA,bA,sm['output'])
    if path=='translation':
        V=V*k;V[:,1]=np.minimum(V[:,1],V[0,1]);V[:,0]=V[0,0];V[:,2]=V[0,2]
    samps.append({'input':addacc(T,'SCALAR',True),'output':addacc(V,'VEC4' if path=='rotation' else 'VEC3'),'interpolation':sm.get('interpolation','LINEAR')})
    chans.append({'sampler':len(samps)-1,'target':{'node':nodeN[nm],'path':path}})
j['animations']=[a for a in j['animations'] if a.get('name')!='Jumping']+[{'name':'Jumping','channels':chans,'samplers':samps}]
while len(b)%4:b.append(0)
j['buffers'][0]['byteLength']=len(b)
js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
open(F,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(b))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(b),0x004E4942)+bytes(b))
print('Jumping',len(chans),'canais, duração',float(max(acc(j,b,s['input']).max() for s in samps)))
