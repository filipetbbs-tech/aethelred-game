"""Meia capa (Meshy) para o herói: grava capa.glb com as texturas reduzidas (1024 px, JPEG) para caber no jogo e no link.
O jogo prende a capa atrás da ombreira e a anima (ver CAPA no index.html).
Uso: python3 ferramentas/capa.py capa_do_meshy.glb"""
import os,sys,json,struct,io
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d=open(sys.argv[1],'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b=d[28+l:]
nb=bytearray();views=[]
for vi,vw in enumerate(j['bufferViews']):
    o=vw.get('byteOffset',0);data=b[o:o+vw['byteLength']]
    img=[k for k,im in enumerate(j['images']) if im['bufferView']==vi]
    if img:
        im=Image.open(io.BytesIO(data)).convert('RGB').resize((1024,1024),Image.LANCZOS);bio=io.BytesIO();im.save(bio,'JPEG',quality=88);data=bio.getvalue()
        j['images'][img[0]]['mimeType']='image/jpeg'
    while len(nb)%4:nb.append(0)
    nv=dict(vw);nv['byteOffset']=len(nb);nv['byteLength']=len(data);nb.extend(data);views.append(nv)
while len(nb)%4:nb.append(0)
j['bufferViews']=views;j['buffers']=[{'byteLength':len(nb)}]
js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
out=os.path.join(ROOT,'capa.glb');open(out,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(nb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(nb),0x004E4942)+bytes(nb))
print(out,os.path.getsize(out))
