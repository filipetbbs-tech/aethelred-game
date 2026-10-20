"""Troca a textura de cor (imagem 0) de um .glb por outra imagem, gravando como JPEG.
Uso: python3 ferramentas/trocar_textura.py modelo.glb nova_textura.png [qualidade]"""
import sys,json,struct,io
from PIL import Image
F,img=sys.argv[1:3];q=int(sys.argv[3]) if len(sys.argv)>3 else 92
d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b=bytearray(d[28+l:])
bio=io.BytesIO();Image.open(img).convert('RGB').save(bio,'JPEG',quality=q);data=bio.getvalue()
while len(b)%4:b.append(0)
j['bufferViews'].append({'buffer':0,'byteOffset':len(b),'byteLength':len(data)});b.extend(data)
old=j['images'][0]['bufferView'];j['images'][0]['bufferView']=len(j['bufferViews'])-1;j['images'][0]['mimeType']='image/jpeg'
# compacta: tira do buffer a imagem antiga se ninguém mais usa
used={a['bufferView'] for a in j['accessors'] if 'bufferView' in a}|{im['bufferView'] for im in j['images']}
nb=bytearray();remap={};views=[]
for i,v in enumerate(j['bufferViews']):
    if i not in used:continue
    while len(nb)%4:nb.append(0)
    o=v.get('byteOffset',0);v=dict(v);seg=b[o:o+v['byteLength']];v['byteOffset']=len(nb);nb.extend(seg);remap[i]=len(views);views.append(v)
while len(nb)%4:nb.append(0)
for a in j['accessors']:
    if 'bufferView' in a:a['bufferView']=remap[a['bufferView']]
for im in j['images']:im['bufferView']=remap[im['bufferView']]
j['bufferViews']=views;j['buffers'][0]['byteLength']=len(nb)
js=json.dumps(j,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
open(F,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(nb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(nb),0x004E4942)+bytes(nb))
print(F,28+len(js)+len(nb))
