"""Funções comuns dos scripts de rig: leitura de GLB e o esqueleto de aethelred_rig_leve.glb (nomes dos ossos, matrizes
de ligação IB, posições das juntas BP no espaço de ligação). A malha desse arquivo não é mais usada no jogo."""
import os,sys,json,struct,io,base64,numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rr
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nrm=np.linalg.norm
def loadglb(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b0=20+l+8
    def acc(i):
        a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=b0+bv.get('byteOffset',0)+a.get('byteOffset',0)
        ct={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32,5122:np.int16,5120:np.int8}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        es=np.dtype(ct).itemsize*n;st=bv.get('byteStride',es)
        if st==es:x=np.frombuffer(d,dtype=ct,count=a['count']*n,offset=off).reshape(a['count'],n)
        else:
            raw=np.frombuffer(d,dtype=np.uint8,count=st*(a['count']-1)+es,offset=off)
            x=np.lib.stride_tricks.as_strided(raw,shape=(a['count'],es),strides=(st,1)).copy().view(ct).reshape(a['count'],n)
        if a.get('normalized'):x=x.astype(np.float32)/np.iinfo(ct).max
        return x
    return d,j,b0,acc
def imgbytes(d,j,b0,k):
    img=j['images'][k];bv=j['bufferViews'][img['bufferView']];o=b0+bv.get('byteOffset',0);return d[o:o+bv['byteLength']],img.get('mimeType','image/jpeg')
def sample(tex,uv):
    H,Wd=tex.shape[:2];return tex[np.clip((uv[:,1]%1)*H,0,H-1).astype(int),np.clip((uv[:,0]%1)*Wd,0,Wd-1).astype(int)]
# --- modelo atual (rigado) ---
d,j,b0,acc=loadglb(os.path.join(ROOT,'aethelred_rig_leve.glb'));p=j['meshes'][0]['primitives'][0]
P=acc(p['attributes']['POSITION']).astype(np.float64);J=acc(p['attributes']['JOINTS_0']).astype(int);W=acc(p['attributes']['WEIGHTS_0']).astype(np.float64)
UVo=acc(p['attributes']['TEXCOORD_0'])
sk=j['skins'][0];names=[j['nodes'][i]['name'].replace('mixamorig','') for i in sk['joints']];bi={n:i for i,n in enumerate(names)}
IB=acc(sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1).astype(np.float64);BW=np.linalg.inv(IB);BP=BW[:,:3,3]
Co=sample(np.asarray(Image.open(io.BytesIO(imgbytes(d,j,b0,0)[0])).convert('RGB')).astype(np.float64)/255,UVo)
def ws(keys):
    w=np.zeros(len(P))
    for k in keys:w+=(W*(J==bi[k])).sum(1)
    return w
armBones=[n for n in names if any(s in n for s in ['Arm','Hand']) and 'Shoulder' not in n]
armw=ws(armBones)
