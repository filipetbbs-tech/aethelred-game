"""Queixo do personagem do Mixamo (aethelred_personagem.glb): o rosto e o queixo estavam com 30-50% do peso na
coluna (Spine2), então no idle o queixo esticava até o peito. Aqui o rosto e o queixo ficam 100% no osso da cabeça,
e a pele do pescoço faz a transição até o peito. Uso: python3 ferramentas/queixo_mixamo.py (só uma vez)"""
import os,json,struct,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F=os.path.join(ROOT,'aethelred_personagem.glb')
d=bytearray(open(F,'rb').read());l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);B0=28+l
CT={5126:np.float32,5123:np.uint16,5121:np.uint8};NC={'VEC3':3,'VEC4':4}
def view(i):
    a=j['accessors'][i];v=j['bufferViews'][a['bufferView']];dt=np.dtype(CT[a['componentType']]);n=NC[a['type']]
    st=v.get('byteStride',dt.itemsize*n);off=B0+v.get('byteOffset',0)+a.get('byteOffset',0)
    return np.ndarray((a['count'],n),dt,buffer=d,offset=off,strides=(st,dt.itemsize)),a
at=j['meshes'][0]['primitives'][0]['attributes']
P,_=view(at['POSITION']);J,_=view(at['JOINTS_0']);W,wa=view(at['WEIGHTS_0'])
names=[j['nodes'][n]['name'].replace('mixamorig:','') for n in j['skins'][0]['joints']]
HEAD=names.index('Head')
Wf=W.astype(np.float64)/(255. if W.dtype==np.uint8 else 65535. if W.dtype==np.uint16 else 1.)
x,y,z=P[:,0],P[:,1],P[:,2]
sm=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
t=sm((y-0.01480)/0.00050)                       # 0 no peito, 1 do queixo para cima
t*=sm((0.00088-np.abs(x))/0.00020)              # só a faixa do rosto/pescoço (não pega o capuz dos lados)
t*=sm((z+0.0002)/0.0003)                        # só a frente
cnt=0
for i in np.nonzero(t>0.001)[0]:
    s=np.nonzero(J[i]==HEAD)[0]
    if len(s)==0:k=int(np.argmin(Wf[i]));J[i,k]=HEAD;Wf[i,k]=0;s=[k]
    oh=np.zeros(4);oh[s[0]]=1;w=Wf[i]*(1-t[i])+oh*t[i];Wf[i]=w/w.sum();cnt+=1
if W.dtype==np.uint8:W[:]=np.round(Wf*255)
elif W.dtype==np.uint16:W[:]=np.round(Wf*65535)
else:W[:]=Wf
open(F,'wb').write(d);print('vértices ajustados',cnt)
