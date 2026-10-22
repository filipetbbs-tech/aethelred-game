"""Cabeça nova (busto do Meshy) no personagem do Mixamo.
1) Marca em extras.headTri de aethelred_personagem.glb os triângulos da pele do rosto (bits em base64, um por
   triângulo): o jogo os esconde e põe a cabeça nova no lugar, dentro do capuz.
2) Grava cabeca.glb: o busto com as texturas reduzidas (para caber no jogo e no link).
Uso: python3 ferramentas/cabeca.py busto_do_meshy.glb"""
import os,sys,json,struct,io,base64,numpy as np
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def rd(f):
    d=open(f,'rb').read();l=struct.unpack('<I',d[12:16])[0];return json.loads(d[20:20+l]),d[28+l:]
def wr(f,j,b):
    js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4);b=bytes(b);b+=b'\0'*((4-len(b)%4)%4)
    open(f,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(b))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(b),0x004E4942)+b)
# --- 1) pele do rosto ---
F=os.path.join(ROOT,'aethelred_personagem.glb');j,b=rd(F)
def acc(i):
    a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];ct={5126:np.float32,5121:np.uint8,5123:np.uint16,5125:np.uint32}[a['componentType']]
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']];it=np.dtype(ct).itemsize
    return np.ndarray((a['count'],n),ct,buffer=b,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',it*n),it)).copy()
pr=j['meshes'][0]['primitives'][0];P=acc(pr['attributes']['POSITION']).astype(float);UV=acc(pr['attributes']['TEXCOORD_0']).astype(float)
I=acc(pr['indices']).reshape(-1,3).astype(int)
v=j['bufferViews'][j['images'][0]['bufferView']];o=v.get('byteOffset',0);T=np.asarray(Image.open(io.BytesIO(b[o:o+v['byteLength']])).convert('RGB'),float)
H,Wd=T.shape[:2]
def samp(uv):return T[np.clip((uv[:,1]*H).astype(int),0,H-1),np.clip((uv[:,0]*Wd).astype(int),0,Wd-1)]
c=(samp(UV[I[:,0]])+samp(UV[I[:,1]])+samp(UV[I[:,2]])+samp(UV[I].mean(1)))/4;cen=P[I].mean(1)
skin=(cen[:,1]>0.0150)&(np.abs(cen[:,0])<0.0012)&(c[:,0]>95)&(c[:,0]-c[:,2]>20)&(c[:,0]>c[:,1])
print('triângulos do rosto',int(skin.sum()))
j['meshes'][0].setdefault('extras',{})['headTri']=base64.b64encode(np.packbits(skin.astype(np.uint8),bitorder='little').tobytes()).decode()
wr(F,j,b)
# --- 2) busto com texturas menores ---
if len(sys.argv)>1:
    j2,b2=rd(sys.argv[1]);nb=bytearray();views=[]
    sizes={0:1024,1:512,2:1024}
    for vi,vw in enumerate(j2['bufferViews']):
        o=vw.get('byteOffset',0);data=b2[o:o+vw['byteLength']]
        img=[k for k,im in enumerate(j2['images']) if im['bufferView']==vi]
        if img:
            im=Image.open(io.BytesIO(data)).convert('RGB');s=sizes.get(img[0],1024);im=im.resize((s,s),Image.LANCZOS)
            bio=io.BytesIO();im.save(bio,'JPEG',quality=88);data=bio.getvalue()
        while len(nb)%4:nb.append(0)
        nv=dict(vw);nv['byteOffset']=len(nb);nv['byteLength']=len(data);nb.extend(data);views.append(nv)
    j2['bufferViews']=views;j2['buffers']=[{'byteLength':len(nb)}]
    wr(os.path.join(ROOT,'cabeca.glb'),j2,nb);print('cabeca.glb',os.path.getsize(os.path.join(ROOT,'cabeca.glb')))
