"""Viking do Mixamo (Meshy): escudo e machado viram peças rígidas (100% num osso: escudo no antebraço esquerdo, machado na
mão direita) e as animações do pack Sword and Shield entram no mesmo GLB. Feito para os arquivos convertidos com FBX2glTF."""
import json,struct,numpy as np,glob,os,re
from PIL import Image
SC='/tmp/claude-0/-home-user-aethelred-game/f693d26f-a336-56f4-a564-05937b894639/scratchpad/'
f=SC+'pack_glb/viking_para_mixamo.glb'
d=open(f,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b=bytearray(d[28+l:])
def accj(jj,bb,i):
    a=jj['accessors'][i];v=jj['bufferViews'][a['bufferView']];ct={5126:np.float32,5125:np.uint32,5123:np.uint16,5121:np.uint8}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    it=np.dtype(ct).itemsize;return np.ndarray((a['count'],n),ct,buffer=bytes(bb),offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',it*n),it)).copy()
acc=lambda i:accj(j,b,i)
p=j['meshes'][0]['primitives'][0];P=acc(p['attributes']['POSITION']).astype(float)
names=[j['nodes'][k]['name'].replace('mixamorig:','') for k in j['skins'][0]['joints']]
IB=acc(j['skins'][0]['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
s=P[:,1].max()/1.9;Pm=P/s
jp=lambda n:np.linalg.inv(IB[names.index(n)])[:3,3]/s
def axis(A,B):
    d=B-A;L=np.linalg.norm(d);d/=L;t=(Pm-A)@d/L;r=np.linalg.norm(Pm-A-np.outer(t*L,d),axis=1);return t,r
weapon={}
# escudo (lado esquerdo): plano do disco ajustado nos pontos longe do braço
E,H=jp('LeftForeArm'),jp('LeftHand');t,r=axis(E,H);side=Pm[:,0]>E[0]-.02
far=side&(r>.09);C=Pm[far].mean(0);u,sv,vt=np.linalg.svd(Pm[far]-C);n=vt[2]
if (H-C)@n>0:n=-n          # normal para fora (lado oposto ao braço)
dist=(Pm-C)@n;rad=np.linalg.norm((Pm-C)-np.outer(dist,n),axis=1);R=np.percentile(rad[far],99)
shield=side&(((r>.09))|((dist>-.025)&(rad<R*1.02)&(np.abs(dist)<.09)))
weapon['LeftForeArm']=shield
# machado (lado direito): linha do cabo ajustada nos pontos longe do braço que formam o cabo
E,H=jp('RightForeArm'),jp('RightHand');t,r=axis(E,H);side=Pm[:,0]<E[0]+.02
far=side&(r>.09)
Q=Pm[far];C2=Q.mean(0);u,sv,vt=np.linalg.svd(Q-C2);dl=vt[0]
tt=(Pm-C2)@dl;rr=np.linalg.norm(Pm-C2-np.outer(tt,dl),axis=1)
axe=side&((r>.09)|((rr<.035)&(tt>=(Q-C2)@dl.min() if False else (tt>((Q-C2)@dl).min()-.02))&(tt<((Q-C2)@dl).max()+.02)))
weapon['RightHand']=axe
print('escudo',shield.sum(),'machado',axe.sum(),'R',round(R,3))
# pesos rígidos
J=acc(p['attributes']['JOINTS_0']).copy();W=acc(p['attributes']['WEIGHTS_0']).astype(np.float32)
for bone,m in weapon.items():
    J[m]=0;J[m,0]=names.index(bone);W[m]=0;W[m,0]=1
def setacc(i,arr):
    a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];o=v.get('byteOffset',0)+a.get('byteOffset',0)
    raw=arr.astype({5126:np.float32,5125:np.uint32,5123:np.uint16,5121:np.uint8}[a['componentType']]).tobytes()
    assert v.get('byteStride',len(raw)//a['count'])==len(raw)//a['count'];b[o:o+len(raw)]=raw
wa=j['accessors'][p['attributes']['WEIGHTS_0']]
if wa['componentType']!=5126: W=np.round(W*(255 if wa['componentType']==5121 else 65535))
setacc(p['attributes']['JOINTS_0'],J);setacc(p['attributes']['WEIGHTS_0'],W)
# animações do pack
nodeN={n.get('name',''):i for i,n in enumerate(j['nodes'])}
def addacc(arr,typ,mm=False):
    while len(b)%4:b.append(0)
    o=len(b);b.extend(arr.astype(np.float32).tobytes());j['bufferViews'].append({'buffer':0,'byteOffset':o,'byteLength':arr.nbytes})
    a={'bufferView':len(j['bufferViews'])-1,'componentType':5126,'count':len(arr),'type':typ}
    if mm:a['min']=[float(arr.min())];a['max']=[float(arr.max())]
    j['accessors'].append(a);return len(j['accessors'])-1
j['animations']=[]
for af in sorted(glob.glob(SC+'pack_glb/*.glb')):
    nm=os.path.basename(af)[:-4]
    if nm=='viking_para_mixamo':continue
    dd=open(af,'rb').read();ll=struct.unpack('<I',dd[12:16])[0];ja=json.loads(dd[20:20+ll]);ba=dd[28+ll:]
    if not ja.get('animations'):print('sem animação',nm);continue
    an=ja['animations'][0];ch=[];sm=[]
    for c in an['channels']:
        tn=ja['nodes'][c['target']['node']].get('name','');pth=c['target']['path']
        if tn not in nodeN or pth=='scale' or (pth=='translation' and tn!='mixamorig:Hips'):continue
        S=an['samplers'][c['sampler']];T=accj(ja,ba,S['input']).ravel();V=accj(ja,ba,S['output']).astype(np.float32)
        sm.append({'input':addacc(T,'SCALAR',True),'output':addacc(V,'VEC4' if pth=='rotation' else 'VEC3'),'interpolation':S.get('interpolation','LINEAR')})
        ch.append({'sampler':len(sm)-1,'target':{'node':nodeN[tn],'path':pth}})
    nice=re.sub(r'_+',' ',nm).strip()
    j['animations'].append({'name':nice,'channels':ch,'samplers':sm})
while len(b)%4:b.append(0)
j['buffers'][0]['byteLength']=len(b)
js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
out=SC+'viking_rig.glb';open(out,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(b))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(b),0x004E4942)+bytes(b))
print('animações',[a['name'] for a in j['animations']],os.path.getsize(out))
np.save(SC+'vk_weapon.npy',np.stack([shield,axe]))
