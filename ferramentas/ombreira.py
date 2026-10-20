"""Marca os vértices da ombreira de couro (ombro direito) de aethelred_personagem.glb em extras.pauldron (bits em
base64, como extras.armHide). Na transformação o jogo mantém a ombreira visível e a aumenta em volta do ombro para
cobrir o braço mutante. Uso: python3 ferramentas/ombreira.py (uma vez)"""
import os,json,struct,io,base64,numpy as np
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));F=os.path.join(ROOT,'aethelred_personagem.glb')
d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b=d[28+l:]
def acc(i):
    a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];ct={5126:np.float32,5121:np.uint8,5123:np.uint16,5125:np.uint32}[a['componentType']]
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']];it=np.dtype(ct).itemsize
    return np.ndarray((a['count'],n),ct,buffer=b,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',it*n),it)).copy()
pr=j['meshes'][0]['primitives'][0];P=acc(pr['attributes']['POSITION']).astype(float);UV=acc(pr['attributes']['TEXCOORD_0']).astype(float)
names=[j['nodes'][n]['name'].replace('mixamorig:','') for n in j['skins'][0]['joints']]
IB=acc(j['skins'][0]['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1);A=np.linalg.inv(IB[names.index('RightArm')])[:3,3]
v=j['bufferViews'][j['images'][0]['bufferView']];o=v.get('byteOffset',0);T=np.asarray(Image.open(io.BytesIO(b[o:o+v['byteLength']])).convert('RGB'),float)
col=T[np.clip((UV[:,1]*T.shape[0]).astype(int),0,T.shape[0]-1),np.clip((UV[:,0]*T.shape[1]).astype(int),0,T.shape[1]-1)]
sel=(np.linalg.norm(P-A,axis=1)<0.0016)&(P[:,0]<-0.0011)&(col.max(1)<85)&(P[:,1]>A[1]-0.0005)
print('vértices da ombreira',int(sel.sum()))
j['meshes'][0].setdefault('extras',{})['pauldron']=base64.b64encode(np.packbits(sel.astype(np.uint8),bitorder='little').tobytes()).decode()
js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
open(F,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(b))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(b),0x004E4942)+b);print('ok')
