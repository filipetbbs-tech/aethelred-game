"""Devolve ao aethelred_personagem.glb os mapas do Meshy perdidos na conversão do Mixamo (FBX Phong -> metal .4 / rugosidade .3,
pele reluzente): rugosidade/metal e normal map de modelos/rogue_warrior_t_pose_leve.glb (mesmo UV e mesma textura de cor).
Uso: python3 ferramentas/material_meshy.py"""
import os,json,struct
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def rd(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);bl=struct.unpack('<I',d[20+l:24+l])[0];return j,bytearray(d[28+l:28+l+bl])
F=os.path.join(ROOT,'aethelred_personagem.glb')
j,b=rd(F); jm,bm=rd(os.path.join(ROOT,'modelos/rogue_warrior_t_pose_leve.glb'))
mm=jm['materials'][0]
def img_of(tex_index):
    im=jm['images'][jm['textures'][tex_index]['source']]; v=jm['bufferViews'][im['bufferView']]; o=v.get('byteOffset',0)
    return bytes(bm[o:o+v['byteLength']]),im.get('mimeType','image/jpeg')
def add_tex(data,mime):
    while len(b)%4:b.append(0)
    j['bufferViews'].append({'buffer':0,'byteOffset':len(b),'byteLength':len(data)}); b.extend(data)
    j['images'].append({'mimeType':mime,'bufferView':len(j['bufferViews'])-1})
    j['textures'].append({'source':len(j['images'])-1,'sampler':j['textures'][0].get('sampler',0)}); return len(j['textures'])-1
mat=j['materials'][0]
assert 'normalTexture' not in mat, 'já aplicado'
mr=add_tex(*img_of(mm['pbrMetallicRoughness']['metallicRoughnessTexture']['index']))
nm=add_tex(*img_of(mm['normalTexture']['index']))
mat.pop('extras',None)
mat['pbrMetallicRoughness'].update({'metallicFactor':1.0,'roughnessFactor':1.0,'metallicRoughnessTexture':{'index':mr}})
mat['normalTexture']={'index':nm}
while len(b)%4:b.append(0)
j['buffers'][0]['byteLength']=len(b)
js=json.dumps(j,separators=(',',':')).encode()
while len(js)%4:js+=b' '
out=struct.pack('<III',0x46546C67,2,28+len(js)+len(b))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(b),0x004E4942)+bytes(b)
open(F,'wb').write(out);print(F,len(out))
