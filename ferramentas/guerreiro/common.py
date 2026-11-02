import bpy, numpy as np, mathutils, os
E = lambda k, d: float(os.environ.get(k, d))

def coords(o):
    me = o.data; co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); return co.reshape(-1, 3)

def edges(o):
    ed = np.empty(len(o.data.edges) * 2, np.int32); o.data.edges.foreach_get('vertices', ed); return ed.reshape(-1, 2)

def components(n, ed, mask):
    # rótulo de componente conexo (só entre vértices da máscara); -1 fora da máscara
    par = np.arange(n)
    def find(i):
        r = i
        while par[r] != r: r = par[r]
        while par[i] != r: par[i], i = r, par[i]
        return r
    for a, b in ed[mask[ed[:, 0]] & mask[ed[:, 1]]]:
        ra, rb = find(a), find(b)
        if ra != rb: par[max(ra, rb)] = min(ra, rb)
    lab = np.full(n, -1); idx = np.where(mask)[0]; lab[idx] = [find(i) for i in idx]; return lab

def nose(co, z0, z1):
    m = (np.abs(co[:, 0]) < 0.01) & (co[:, 2] > z0) & (co[:, 2] < z1); i = np.where(m)[0][np.argmin(co[m, 1])]; return co[i]

def head_matrix(cb, ch):
    nb, nh = nose(cb, 0.86, 0.89), nose(ch, 0.48, 0.55)
    return (mathutils.Matrix.Translation(mathutils.Vector(nb) + mathutils.Vector((0, E('DY', 0), E('DZ', 0))))
            @ mathutils.Matrix.Rotation(np.radians(E('RX', 0)), 4, 'X') @ mathutils.Matrix.Scale(E('S', 0.2), 4)
            @ mathutils.Matrix.Translation(-mathutils.Vector(nh))), nb

def hair_mask(body, cb, D):
    # cabelo = albedo escuro/saturado em componentes grandes + ilhas soltas da malha na cabeça + buracos pequenos preenchidos
    vc = np.load(f'{D}/vc_corpo.npy'); lum = vc @ np.array([.299, .587, .114]); sat = (vc.max(1) - vc.min(1)) / np.maximum(vc.max(1), 1e-4)
    n = len(cb); ed = edges(body); zone = cb[:, 2] > E('HZ', 0.5)
    hair = (lum < E('HL', 0.40)) & (sat > E('HS', 0.30)) & zone
    lab = components(n, ed, hair); u, c = np.unique(lab[hair], return_counts=True)
    keep = np.isin(lab, u[c >= E('MINC', 2000)]) & hair
    # ilhas soltas (fios separados) acima do queixo e fora do tronco
    isl = components(n, ed, np.ones(n, bool)); u, c = np.unique(isl, return_counts=True)
    small = u[c < E('ISL', 5000)]
    loose = np.isin(isl, small) & (cb[:, 2] > 0.80)
    print('ilhas: maiores', np.sort(c)[-6:], '| soltas na cabeça', len(np.unique(isl[loose])), 'vértices', loose.sum())
    for k in np.unique(isl[loose])[:25]:
        p = cb[isl == k]; print('  ilha', len(p), 'centro', p.mean(0).round(3), 'z', p[:, 2].min().round(3), p[:, 2].max().round(3))
    keep |= loose
    # buracos: componentes pequenos do complemento dentro da zona viram cabelo
    rest = zone & ~keep; lab = components(n, ed, rest); u, c = np.unique(lab[rest], return_counts=True)
    keep |= np.isin(lab, u[c < E('HOLE', 3000)]) & rest
    return keep
