# monta aethelred_guerreiro.glb: o esqueleto, as animações e a estrutura de aethelred_personagem.glb, com a malha, os
# pesos e a textura (atlas) do guerreiro. Só as translações dos ossos mudam (rotações de repouso iguais às do Mixamo, então
# as animações valem sem alteração); a translação do quadril nas animações é ajustada pela altura nova do quadril.
import json, struct, os, sys, base64, numpy as np
D = os.path.dirname(__file__)
OLD = r'C:\Jogos\aethelred-game\aethelred_personagem.glb'; OUTF = os.environ.get('OUTF', r'C:\Jogos\aethelred-game\aethelred_guerreiro.glb')
exec(open(os.path.join(D, 'rig.py'), encoding='utf-8').read().split('# --- guerreiro')[0].split('jO, bO = readglb(OLD)')[0])   # readglb, acc, qm, nodeM
j, b = readglb(OLD); b = bytes(b); R = np.load(os.path.join(D, 'rig_out', 'guerreiro_rig.npz'))
pO = acc(j, b, j['meshes'][0]['primitives'][0]['attributes']['POSITION']); ymin = float(pO[:, 1].min())
JP = {str(n): p + np.array([0, ymin, 0]) for n, p in zip(R['JNAMES'], R['JPOS'])}
P = R['P'].astype(np.float64); P[:, 1] += ymin
par = {c: i for i, n in enumerate(j['nodes']) for c in n.get('children', [])}
nm = lambda s: s.replace('mixamorig:', '')
def W(k, nodes):
    M = np.eye(4)
    while k is not None: M = nodeM(nodes[k]) @ M; k = par.get(k)
    return M
Wold = {i: W(i, j['nodes']) for i in range(len(j['nodes']))}
hips = [i for i, n in enumerate(j['nodes']) if n.get('name') == 'mixamorig:Hips'][0]; hy0 = j['nodes'][hips]['translation'][1]
# translações novas, de pai para filho (o mundo do pai já com a translação nova)
order = []; stack = [k for k in j['scenes'][0]['nodes']]
while stack: k = stack.pop(0); order.append(k); stack += j['nodes'][k].get('children', [])
Wnew = {}
for k in order:
    n = j['nodes'][k]; p = par.get(k); Wp = Wnew[p] if p is not None else np.eye(4)
    name = nm(n.get('name', ''))
    if name in JP:
        t = np.linalg.inv(Wp) @ np.r_[JP[name], 1]; n['translation'] = [float(x) for x in t[:3]]
    Wnew[k] = Wp @ nodeM(n)
hy1 = j['nodes'][hips]['translation'][1]; kh = hy1 / hy0
err = max(np.abs(Wnew[i][:3, :3] - Wold[i][:3, :3]).max() for i in Wnew)
print('QUADRIL altura', round(hy0, 5), '->', round(hy1, 5), '(x', round(kh, 3), ') | rotações de repouso mudaram no máx.', err)

blobs = []; views = []; accs = []
def view(data, target=None):
    data = bytes(data); off = sum(len(x) for x in blobs); blobs.append(data + b'\0' * ((4 - len(data) % 4) % 4))
    v = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data)}
    if target: v['target'] = target
    views.append(v); return len(views) - 1
def add(arr, typ, ct, mm=False, target=None):
    a = {'bufferView': view(np.ascontiguousarray(arr).tobytes(), target), 'componentType': ct, 'count': len(arr), 'type': typ}
    if mm: a['min'] = np.atleast_2d(arr).reshape(len(arr), -1).min(0).tolist(); a['max'] = np.atleast_2d(arr).reshape(len(arr), -1).max(0).tolist()
    accs.append(a); return len(accs) - 1
def copyacc(i):   # copia um acessor antigo (dados compactados num bufferView novo)
    a = dict(j['accessors'][i]); d = acc(j, b, i)
    for key in ('min', 'max'):
        if key in a and not isinstance(a[key], list): a[key] = [a[key]]
    a['bufferView'] = view(d.tobytes()); a.pop('byteOffset', None); accs.append(a); return len(accs) - 1

# malha
JNT, WGT = R['JNT'], R['WGT']; jn = [nm(j['nodes'][i]['name']) for i in j['skins'][0]['joints']]
armIdx = [i for i, n in enumerate(jn) if ('Arm' in n or 'Hand' in n) and 'Shoulder' not in n]
aw = (np.isin(JNT, armIdx) * WGT).sum(1)
prim = {'attributes': {'POSITION': add(P.astype(np.float32), 'VEC3', 5126, True, 34962), 'NORMAL': add(R['N'], 'VEC3', 5126, False, 34962),
                       'TEXCOORD_0': add(R['UV'], 'VEC2', 5126, False, 34962), 'JOINTS_0': add(JNT.astype(np.uint16), 'VEC4', 5123, False, 34962),
                       'WEIGHTS_0': add(WGT.astype(np.float32), 'VEC4', 5126, False, 34962)},
        'indices': add(R['IDX'].astype(np.uint32), 'SCALAR', 5125, False, 34963), 'material': 0, 'mode': 4}
j['meshes'][0] = {'name': 'guerreiro', 'primitives': [prim], 'extras': {'armHide': base64.b64encode(np.packbits((aw > .5).astype(np.uint8), bitorder='little').tobytes()).decode()}}
print('MALHA', len(P), 'vértices', len(R['IDX']) // 3, 'triângulos | vértices de braço', int((aw > .5).sum()))
# pele: matrizes de ligação novas (malha na origem)
IB = np.stack([np.linalg.inv(Wnew[i]).T.ravel() for i in j['skins'][0]['joints']]).astype(np.float32)
j['skins'][0]['inverseBindMatrices'] = add(IB, 'MAT4', 5126)
# animações: copia; translação do quadril escalada pela altura nova
for an in j['animations']:
    for ch in an['channels']:
        sm = an['samplers'][ch['sampler']]
        if ch['target']['path'] == 'translation' and ch['target']['node'] == hips and not sm.get('_ok'):
            v = acc(j, b, sm['output']).astype(np.float32) * kh; sm['output'] = add(v, 'VEC3', 5126); sm['input'] = copyacc(sm['input']); sm['_ok'] = 1
    for sm in an['samplers']:
        if not sm.pop('_ok', None): sm['input'] = copyacc(sm['input']); sm['output'] = copyacc(sm['output'])
# material e textura
img = open(os.path.join(D, 'rig_out', 'atlas.jpg'), 'rb').read()
j['images'] = [{'name': 'guerreiro', 'mimeType': 'image/jpeg', 'bufferView': view(img)}]
j['textures'] = [{'name': 'guerreiro', 'sampler': 0, 'source': 0}]
j['materials'] = [{'name': 'pele_guerreiro', 'alphaMode': 'OPAQUE', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0, 'texCoord': 0},
                   'baseColorFactor': [1, 1, 1, 1], 'metallicFactor': 0.0, 'roughnessFactor': 0.62}}]
j['accessors'] = accs; j['bufferViews'] = views; binb = b''.join(blobs); j['buffers'] = [{'byteLength': len(binb)}]
j['asset'] = {'version': '2.0', 'generator': 'aethelred r_monta.py (esqueleto do FBX2glTF/Mixamo)'}
js = json.dumps(j, separators=(',', ':')).encode(); js += b' ' * ((4 - len(js) % 4) % 4)
out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binb)) + struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(binb), 0x004E4942) + binb
open(OUTF, 'wb').write(out); print('GRAVADO', OUTF, round(len(out) / 1e6, 2), 'MB | animações', [a['name'] for a in j['animations']])
