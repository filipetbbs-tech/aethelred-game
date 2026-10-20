import numpy as np, json, struct, io
from PIL import Image
from glb import load, acc
j,b0=load('/home/user/aethelred-game/aethelred_rig_leve.glb'); b0=bytes(b0); b=bytearray(b0)
f=np.load('facesel.npz'); sel,IMG=f['sel'],f['IMG']
pr=j['meshes'][0]['primitives'][0]; at=pr['attributes']
I=acc(j,b0,pr['indices']).copy().ravel(); tris=I.reshape(-1,3)
vid=tris[sel].ravel(); nf=len(vid)
F=Image.open('face_tex.png').convert('RGB')
x0,y0=np.floor(IMG[vid].min(0)).astype(int)-4; x1,y1=np.ceil(IMG[vid].max(0)).astype(int)+4
x0,y0=max(x0,0),max(y0,0); x1,y1=min(x1,F.width),min(y1,F.height)
Fc=F.crop((x0,y0,x1,y1)); print('crop',Fc.size)
uv=np.c_[(IMG[vid,0]-x0)/Fc.width,(IMG[vid,1]-y0)/Fc.height].astype(np.float32)
jb=io.BytesIO(); Fc.save(jb,'JPEG',quality=90); jpg=jb.getvalue(); print('jpg',len(jpg))
def pad(): 
    while len(b)%4: b.append(0)
def addview(data,target=None):
    pad(); off=len(b); b.extend(data); v={'buffer':0,'byteOffset':off,'byteLength':len(data)}
    if target: v['target']=target
    j['bufferViews'].append(v); return len(j['bufferViews'])-1
def addacc(arr,ctype,typ,target=34962,minmax=False):
    bv=addview(arr.tobytes(),target); a={'bufferView':bv,'componentType':ctype,'count':len(arr),'type':typ}
    if minmax: a['min']=arr.min(0).tolist(); a['max']=arr.max(0).tolist()
    j['accessors'].append(a); return len(j['accessors'])-1
CT={np.float32:5126,np.uint16:5123,np.uint8:5121,np.uint32:5125}
nat={}
for k in ['POSITION','NORMAL','JOINTS_0','WEIGHTS_0']:
    a=acc(j,b0,at[k])[vid].copy(); typ=j['accessors'][at[k]]['type']
    nat[k]=addacc(a,j['accessors'][at[k]]['componentType'],typ,minmax=(k=='POSITION'))
    if j['accessors'][at[k]].get('normalized'): j['accessors'][nat[k]]['normalized']=True
nat['TEXCOORD_0']=addacc(uv,5126,'VEC2')
fi=addacc(np.arange(nf,dtype=np.uint32),5125,'SCALAR',34963)
# shrink old indices
rest=tris[~sel].ravel().astype(I.dtype)
ia=j['accessors'][pr['indices']]; iv=j['bufferViews'][ia['bufferView']]; io_=iv.get('byteOffset',0)+ia.get('byteOffset',0)
rb=rest.tobytes(); b[io_:io_+len(rb)]=rb; ia['count']=len(rest)  # reaproveita o espaço dos índices antigos
imv=addview(jpg); j['images'].append({'mimeType':'image/jpeg','bufferView':imv})
j['textures'].append({'source':len(j['images'])-1,'sampler':0})
mat=json.loads(json.dumps(j['materials'][0])); mat['name']='Rosto'; mat['pbrMetallicRoughness']['baseColorTexture']={'index':len(j['textures'])-1}
j['materials'].append(mat)
j['meshes'][0]['primitives'].append({'attributes':nat,'indices':fi,'material':len(j['materials'])-1,'mode':4})
pad(); j['buffers'][0]['byteLength']=len(b)
js=json.dumps(j,separators=(',',':')).encode()
while len(js)%4: js+=b' '
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(b))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(b),0x004E4942)+bytes(b)
open('/home/user/aethelred-game/aethelred_rig_rosto.glb','wb').write(out); print(len(out))

# compacta: remove bufferViews que nenhum accessor/imagem usa mais
j2=json.loads(js); used=sorted({a['bufferView'] for a in j2['accessors'] if 'bufferView' in a}|{im['bufferView'] for im in j2['images']})
nb=bytearray(); remap={}; nviews=[]
for i in used:
    v=dict(j2['bufferViews'][i]); data=b[v.get('byteOffset',0):v.get('byteOffset',0)+v['byteLength']]
    while len(nb)%4: nb.append(0)
    v['byteOffset']=len(nb); nb.extend(data); remap[i]=len(nviews); nviews.append(v)
while len(nb)%4: nb.append(0)
for a in j2['accessors']:
    if 'bufferView' in a: a['bufferView']=remap[a['bufferView']]
for im in j2['images']: im['bufferView']=remap[im['bufferView']]
j2['bufferViews']=nviews; j2['buffers'][0]['byteLength']=len(nb)
js=json.dumps(j2,separators=(',',':')).encode()
while len(js)%4: js+=b' '
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(nb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(nb),0x004E4942)+bytes(nb)
open('/home/user/aethelred-game/aethelred_rig_rosto.glb','wb').write(out); print('compactado',len(out))
