"""Extrai uma animação de um .glb para um arquivo pequeno (só esqueleto + animação, sem malha nem texturas).
Uso: python3 ferramentas/extrair_animacao.py modelo.glb NomeDaAnimacao saida.glb"""
import sys,json,struct
src,anim,out=sys.argv[1:4]
d=open(src,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b=d[28+l:]
A=[a for a in j['animations'] if a.get('name')==anim]
if not A:sys.exit('animações disponíveis: '+', '.join(a.get('name') for a in j['animations']))
A=json.loads(json.dumps(A[0]))
for n in j['nodes']:n.pop('mesh',None);n.pop('skin',None)
nb=bytearray();views=[];accs=[];amap={}
def take(i):
    if i in amap:return amap[i]
    a=dict(j['accessors'][i]);v=j['bufferViews'][a['bufferView']]
    while len(nb)%4:nb.append(0)
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]*{5126:4,5123:2,5121:1,5125:4}[a['componentType']]*a['count']
    st=v.get('byteOffset',0)+a.pop('byteOffset',0);off=len(nb);nb.extend(b[st:st+n])   # só o trecho deste accessor
    views.append({'buffer':0,'byteOffset':off,'byteLength':n});a['bufferView']=len(views)-1;accs.append(a);amap[i]=len(accs)-1;return amap[i]
for s in A['samplers']:s['input']=take(s['input']);s['output']=take(s['output'])
while len(nb)%4:nb.append(0)
o={'asset':j['asset'],'scene':j.get('scene',0),'scenes':j['scenes'],'nodes':j['nodes'],'animations':[A],
   'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(nb)}]}
js=json.dumps(o,separators=(',',':')).encode()
while len(js)%4:js+=b' '
open(out,'wb').write(struct.pack('<III',0x46546C67,2,28+len(js)+len(nb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(nb),0x004E4942)+bytes(nb))
print(out,28+len(js)+len(nb))
