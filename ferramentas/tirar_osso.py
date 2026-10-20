"""Esconde a ponta cortada do ombro do braço mutante antigo, que aparecia como um osso claro na frente do ombro:
1) o toco de osso (úmero) que sai do corte é achatado sobre ele;
2) a ponta do ombro (o corte e a borda clara em volta) afina em direção ao eixo do braço, para ficar por baixo do colete;
3) na textura, as partes claras (osso, tendão, fibras brancas) da ponta do ombro ganham o tom do músculo em volta.
(Antes: "Tira o toco de osso (úmero) que sai do corte do ombro do braço mutante antigo: os vértices do osso que passam do
plano do corte são achatados sobre ele. Edita aethelred_braco_mutante_antigo.glb no lugar (os dois lados) e depois
é preciso rodar ferramentas/braco_misto.py para o braço misto. Uso: python3 ferramentas/tirar_osso.py (uma vez)"""
import os,json,struct,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def rd(F):
    d=bytearray(open(F,'rb').read());l=struct.unpack('<I',d[12:16])[0];return d,json.loads(d[20:20+l]),28+l
def view(d,j,b0,i):
    a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];ct={5126:np.float32,5121:np.uint8,5125:np.uint32,5123:np.uint16}[a['componentType']]
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];it=np.dtype(ct).itemsize
    return np.ndarray((a['count'],n),ct,buffer=d,offset=b0+v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',it*n),it))
dS,jS,bS=rd(os.path.join(ROOT,'modelos','mutated_arm_claw_leve.glb'))
X=view(dS,jS,bS,jS['meshes'][0]['primitives'][0]['attributes']['POSITION']).astype(float);NS=view(dS,jS,bS,jS['meshes'][0]['primitives'][0]['attributes']['NORMAL']).astype(float)
rect=(X[:,0]>-.19)&(X[:,0]<-.05)&(X[:,1]>.58)&(X[:,1]<.8)&(X[:,2]<-1.06)
face=(NS@np.array([0,0,-1.])>.8)&(X[:,2]<-1.0)&~rect
c=np.linalg.lstsq(np.c_[X[face][:,0],X[face][:,1],np.ones(face.sum())],X[face][:,2],rcond=None)[0]
zp=X[:,0]*c[0]+X[:,1]*c[1]+c[2];bone=rect&(X[:,2]<zp-.004)
Q=X.copy();Q[bone,2]=zp[bone]+.002
Z0,ZL,AFINA=-0.92,0.24,0.45
cz=np.arange(-1.25,-0.5,.01);cc=np.array([X[np.abs(X[:,2]-z)<.02].mean(0) if (np.abs(X[:,2]-z)<.02).sum()>20 else [np.nan]*3 for z in cz])
ok=~np.isnan(cc[:,0]);cxz=np.interp(Q[:,2],cz[ok],cc[ok,0]);cyz=np.interp(Q[:,2],cz[ok],cc[ok,1])
u=np.clip((Z0-Q[:,2])/ZL,0,1);f=1-AFINA*u*u*(3-2*u)
tip=Q[:,2]<Z0;bone|=tip
Q[tip,0]=cxz[tip]+(Q[tip,0]-cxz[tip])*f[tip];Q[tip,1]=cyz[tip]+(Q[tip,1]-cyz[tip])*f[tip]
F=os.path.join(ROOT,'aethelred_braco_mutante_antigo.glb');d,j,b0=rd(F)
P=view(d,j,b0,j['meshes'][0]['primitives'][0]['attributes']['POSITION']);nv=len(X)
A=np.c_[X,np.ones(nv)];M=np.linalg.lstsq(A,P[:nv].astype(float),rcond=None)[0]   # arquivo de origem -> espaço do jogo (lado direito)
print('erro do ajuste',np.abs(A@M-P[:nv]).max(),'vértices do osso',bone.sum())
R=np.c_[Q[bone],np.ones(bone.sum())]@M;idx=np.nonzero(bone)[0]
P[idx]=R;P[nv+idx]=R*np.array([-1,1,1.])
# --- textura: tira o branco da ponta do ombro ---
import io
from PIL import Image, ImageFilter
UVs=view(dS,jS,bS,jS['meshes'][0]['primitives'][0]['attributes']['TEXCOORD_0']).astype(float)
TS=view(dS,jS,bS,jS['meshes'][0]['primitives'][0]['indices']).reshape(-1,3).astype(int)
mi=j['materials'][j['meshes'][0]['primitives'][0].get('material',0)]['pbrMetallicRoughness']['baseColorTexture']['index']
im=j['images'][j['textures'][mi]['source']];bv=j['bufferViews'][im['bufferView']];o=b0+bv.get('byteOffset',0)
tex=Image.open(io.BytesIO(bytes(d[o:o+bv['byteLength']]))).convert('RGB');W_,H_=tex.size
reg=Image.new('L',(W_,H_),0);from PIL import ImageDraw;dr=ImageDraw.Draw(reg)
zz=X[:,2];sel=TS[(zz[TS]<-0.86).all(1)]
for t in sel:dr.polygon([(UVs[k,0]*W_,UVs[k,1]*H_) for k in t],fill=255)
reg=np.asarray(reg.filter(ImageFilter.MaxFilter(5)),np.float32)/255
a=np.asarray(tex,np.float32);mx=a.max(2);mn=a.min(2);sat=(mx-mn)/(mx+1)
pale=np.clip((mx-120)/50,0,1)*np.clip((.42-sat)/.15,0,1)
musc=np.asarray(Image.fromarray(a.astype(np.uint8)).filter(ImageFilter.GaussianBlur(40)),np.float32)
tgt=np.array([128,52,50],np.float32)
k=(pale*reg)[...,None]
out=a*(1-k)+(tgt*(a.mean(2,keepdims=True)/150)*.9+tgt*.1)*k
bb=io.BytesIO();Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save(bb,'JPEG',quality=90);nb_=bb.getvalue()
print('texels clareados no ombro',int((pale*reg>.5).sum()))
# regrava o arquivo com a imagem nova no fim do buffer
l=struct.unpack('<I',d[12:16])[0];binb=bytearray(d[b0:])
while len(binb)%4:binb.append(0)
j['bufferViews'].append({'buffer':0,'byteOffset':len(binb),'byteLength':len(nb_)});binb.extend(nb_);im['bufferView']=len(j['bufferViews'])-1;im['mimeType']='image/jpeg'
while len(binb)%4:binb.append(0)
j['buffers'][0]['byteLength']=len(binb)
js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
d=struct.pack('<III',0x46546C67,2,28+len(js)+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+bytes(binb)
open(F,'wb').write(d);print('ok',F)
