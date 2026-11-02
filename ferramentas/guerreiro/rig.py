# encaixa o esqueleto Mixamo do jogo (aethelred_personagem.glb) no guerreiro novo e calcula os pesos da pele.
# As juntas novas são achadas na malha do guerreiro (cortes da perna/braço/tronco, queixo, mão), usando as proporções do
# esqueleto antigo como guia; os pesos vêm do "bone heat" do Blender num esqueleto auxiliar com os ossos ao longo dos membros.
# Saída: rig_out/guerreiro_rig.npz (malha em unidades do arquivo antigo, Y para cima) + rig_out/atlas.jpg
import bpy, json, struct, sys, os, math, heapq, numpy as np, mathutils
D = os.path.dirname(__file__); OUT = os.path.join(D, 'rig_out'); os.makedirs(OUT, exist_ok=True)
OLD = r'C:\Jogos\aethelred-game\aethelred_personagem.glb'
NEW = os.environ.get('NEWGLB', r'C:\Users\Filipe\Downloads\guerreiro_mixamo\guerreiro.glb')

def readglb(F):
    d = open(F, 'rb').read(); l = struct.unpack('<I', d[12:16])[0]; j = json.loads(d[20:20 + l])
    bl = struct.unpack('<I', d[20 + l:24 + l])[0]; return j, d[28 + l:28 + l + bl]
def acc(j, b, i):
    a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]; off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    ct = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a['componentType']]
    n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
    es = np.dtype(ct).itemsize * n; st = bv.get('byteStride', es)
    if st == es: return np.frombuffer(b, dtype=ct, count=a['count'] * n, offset=off).reshape(a['count'], n)
    raw = np.frombuffer(b, dtype=np.uint8, count=st * (a['count'] - 1) + es, offset=off)
    return np.lib.stride_tricks.as_strided(raw, shape=(a['count'], es), strides=(st, 1)).copy().view(ct).reshape(a['count'], n)
def qm(q):
    x, y, z, w = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
def nodeM(n):
    if 'matrix' in n: return np.array(n['matrix']).reshape(4, 4).T
    M = np.eye(4); M[:3, :3] = qm(n.get('rotation', [0, 0, 0, 1])) @ np.diag(n.get('scale', [1, 1, 1])); M[:3, 3] = n.get('translation', [0, 0, 0]); return M
jO, bO = readglb(OLD)
par = {c: i for i, n in enumerate(jO['nodes']) for c in n.get('children', [])}
def worldM(k):
    M = np.eye(4)
    while k is not None: M = nodeM(jO['nodes'][k]) @ M; k = par.get(k)
    return M
G2B = lambda p: np.array([p[..., 0], -p[..., 2], p[..., 1]]).T if p.ndim > 1 else np.array([p[0], -p[2], p[1]])
nm = lambda s: s.replace('mixamorig:', '')
NODES = {nm(n['name']): i for i, n in enumerate(jO['nodes']) if n.get('name', '').startswith('mixamorig')}
skin = jO['skins'][0]; JN = [nm(jO['nodes'][i]['name']) for i in skin['joints']]
pO = acc(jO, bO, jO['meshes'][0]['primitives'][0]['attributes']['POSITION']).astype(np.float64)
To = pO[:, 1].max() - pO[:, 1].min()

# --- guerreiro (Blender, metros, Z para cima, frente -Y, braços em X) ---
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=NEW, merge_vertices=True)
G = [o for o in bpy.data.objects if o.type == 'MESH'][0]; mw = G.matrix_world.copy(); G.parent = None; G.data.transform(mw); G.matrix_world.identity()
me = G.data; Gv = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', Gv); Gv = Gv.reshape(-1, 3)
Gv[:, 2] -= Gv[:, 2].min(); Tn = Gv[:, 2].max(); me.vertices.foreach_set('co', Gv.ravel()); me.update()
F = Tn / To; k = To / Tn                                   # antigo -> metros do guerreiro e volta
Ov = G2B(pO) * F; Ov[:, 2] -= Ov[:, 2].min()
JO = {n: G2B(worldM(i)[:3, 3]) * F for n, i in NODES.items()}
dz0 = G2B(pO)[:, 2].min() * F
for n in JO: JO[n][2] -= dz0
print('ALTURA antiga', round(To, 5), 'nova', round(Tn, 3), 'm | fator', round(F, 2))

tol = 0.006
def slab(V, ax, val, mask, t=tol):
    m = mask & (np.abs(V[:, ax] - val) < t); P = V[m]
    if len(P) < 4: return None, P
    return (P.min(0) + P.max(0)) / 2, P
def crotch(V, T):
    m = (np.abs(V[:, 0]) < tol) & (V[:, 2] > 0.35 * T) & (V[:, 2] < 0.6 * T); return V[m, 2].min()
def tipx(V, s): return (s * V[:, 0]).max()
crO, crN = crotch(Ov, Tn), crotch(Gv, Tn)
def armz(V, s, tip):
    c, _ = slab(V, 0, s * 0.45 * tip, (s * V[:, 0] > 0) & (V[:, 2] > 0.6 * Tn)); return c[2], c
J = {}
for s, S in ((1, 'Left'), (-1, 'Right')):
    tO, tN = tipx(Ov, s), tipx(Gv, s)
    shO, _ = armz(Ov, s, tO); shN, _ = armz(Gv, s, tN); J['_sh' + S] = (shO, shN, tO, tN)
shO = (J['_shLeft'][0] + J['_shRight'][0]) / 2; shN = (J['_shLeft'][1] + J['_shRight'][1]) / 2
print('VIRILHA antiga', round(crO, 3), 'nova', round(crN, 3), '| linha dos braços', round(shO, 3), round(shN, 3))

# queixo (mento) do guerreiro: maior salto para a frente no perfil da frente da linha do meio (só cortes com pele na frente)
mid = Gv[np.abs(Gv[:, 0]) < 0.012]; zs = np.arange(0.81 * Tn, 0.90 * Tn, 0.003)
fy = np.array([mid[np.abs(mid[:, 2] - z) < 0.003, 1].min() if (np.abs(mid[:, 2] - z) < 0.003).any() else np.nan for z in zs])
ok = fy < -0.01; zv, yv = zs[ok], fy[ok]; i = np.argmin(yv[1:] - yv[:-1]); menton = (zv[i] + zv[i + 1]) / 2
print('QUEIXO z', round(menton, 3), '| salto', round(yv[i + 1] - yv[i], 3))

# --- tronco: quadril pela junta da coxa; coluna entre o quadril e o pescoço nas proporções do esqueleto antigo ---
def torso_y(V, z):
    c, P = slab(V, 2, z, np.abs(V[:, 0]) < 0.06 * Tn / 1.9)
    return P[:, 1].min(), P[:, 1].max()
UPZ = crN + float(os.environ.get('UP', 0.06)) * Tn
nc, _ = slab(Gv, 2, menton - 0.03, np.abs(Gv[:, 0]) < 0.08)
J['Head'] = np.array([0.0, nc[1] + 0.004, menton + 0.012])
fN = (JO['Neck'][2] - shO) / (JO['Head'][2] - shO); zn = shN + fN * (J['Head'][2] - shN)
c, _ = slab(Gv, 2, zn, np.abs(Gv[:, 0]) < 0.08); J['Neck'] = np.array([0.0, c[1], zn])
J['HeadTop_End'] = np.array([0.0, J['Head'][1], Tn])
hipN = UPZ + (JO['Hips'][2] - JO['LeftUpLeg'][2])
for n in ('Hips', 'Spine', 'Spine1', 'Spine2'):
    zo = JO[n][2]; zn = hipN + (zo - JO['Hips'][2]) * (J['Neck'][2] - hipN) / (JO['Neck'][2] - JO['Hips'][2])
    a, b = torso_y(Ov, zo); f = np.clip((JO[n][1] - a) / (b - a), 0.3, 0.75); a2, b2 = torso_y(Gv, zn)
    J[n] = np.array([0.0, a2 + f * (b2 - a2), zn])
if os.environ.get('DRY'):
    for z, y in zip(zs[::5], fy[::5]): print(f'   perfil {z:.3f} {y:+.3f}')

# --- braços e mãos ---
CL = {}                                       # linha do centro de cada dedo (pontos, distância das juntas ao longo dela)
for s, S in ((1, "Left"), (-1, "Right")):
    _, _, tO, tN = J['_sh' + S]; r = tN / tO
    am = (s * Gv[:, 0] > 0) & (Gv[:, 2] > crN)
    xa, xf, xh = JO[S + 'Arm'][0] * r, JO[S + 'ForeArm'][0] * r, JO[S + 'Hand'][0] * r
    cf, _ = slab(Gv, 0, xf, am); ch, _ = slab(Gv, 0, xh, am)
    x1 = xa + 0.35 * (xf - xa); c1, _ = slab(Gv, 0, x1, am)
    ca = c1 + (c1 - cf) * (xa - x1) / (x1 - xf)
    J[S + 'Arm'] = np.array([xa, ca[1], ca[2]]); J[S + 'ForeArm'] = np.array([xf, cf[1], cf[2]]); J[S + 'Hand'] = np.array([xh, ch[1], ch[2]])
    so, ao = JO[S + 'Shoulder'], JO[S + 'Arm']; J[S + 'Shoulder'] = np.array([so[0] * J[S + 'Arm'][0] / ao[0], J[S + 'Arm'][1] + so[1] - ao[1], J[S + 'Arm'][2] + so[2] - ao[2]])
    dO = JO[S + 'ForeArm'] - JO[S + 'Arm']; dN = J[S + 'ForeArm'] - J[S + 'Arm']
    print(S, 'BRAÇO inclinação antiga', round(math.degrees(math.atan2(dO[2], abs(dO[0]))), 1), 'nova', round(math.degrees(math.atan2(dN[2], abs(dN[0]))), 1), 'graus')
    # pulso: o ponto mais estreito do antebraço antes de a mão alargar (pela proporção do esqueleto antigo ficava ~4 cm para dentro)
    E_ = J[S + 'ForeArm']; l0 = np.linalg.norm(J[S + 'Hand'] - E_); ax = (J[S + 'Hand'] - E_) / l0; u0 = (Gv - E_) @ ax
    ED = np.empty(len(me.edges) * 2, np.int64); me.edges.foreach_get('vertices', ED); ED = ED.reshape(-1, 2)
    def corte(q):   # contorno exato: arestas que atravessam o plano perpendicular ao antebraço
        a, b = u0[ED[:, 0]] - q, u0[ED[:, 1]] - q; e = ED[(a * b < 0) & am[ED[:, 0]]]; t = (u0[e[:, 0]] - q) / (u0[e[:, 0]] - u0[e[:, 1]])
        P = Gv[e[:, 0]] + t[:, None] * (Gv[e[:, 1]] - Gv[e[:, 0]]); return P[np.linalg.norm(P - P.mean(0), axis=1) < 0.09]
    qs = np.arange(l0 - 0.03, l0 + 0.08, 0.005); th = np.array([np.ptp(P[:, 1]) + np.ptp(P[:, 2]) for P in map(corte, qs)])
    th = np.convolve(th, np.ones(3) / 3, 'same'); th[0] = th[-1] = 1e9; q = qs[np.argmin(th)]; P = corte(q)
    J[S + 'Hand'] = (P.min(0) + P.max(0)) / 2
    print(S, 'PULSO movido', round((q - l0) * 100, 1), 'cm para a mão | antebraço', round(q, 3), 'm')
    # mão: o esqueleto espera a palma para baixo, a fileira dos dedos na horizontal e o polegar para a frente (pose T do
    # Mixamo). Mede a fileira dos 4 dedos na malha do guerreiro e gira o antebraço (torção que cresce do cotovelo ao pulso)
    def polegar(V, wr, ax):
        H = V[(V - wr) @ ax > 0]; u = (H - wr) @ ax; L = u.max()
        pal = H[(u > 0.08 * L) & (u < 0.3 * L)][:, 1:]; r = np.linalg.eigh(np.cov(pal.T))[1][:, 1]   # fileira dos dedos = eixo largo da palma
        pr = H[:, 1:] @ r; ft = pr[u > 0.75 * L]; mid = pr[(u > 0.2 * L) & (u < 0.7 * L)]
        lo, hi = ft.min() - mid.min(), mid.max() - ft.max()                       # o polegar sobra além da fileira dos dedos
        return (r if hi > lo else -r), max(lo, hi)
    E_, W_ = J[S + 'ForeArm'], J[S + 'Hand']; ax = (W_ - E_) / np.linalg.norm(W_ - E_)
    dOld = (JO[S + 'HandIndex1'] - JO[S + 'HandPinky1'])[1:]; dOld = dOld / np.linalg.norm(dOld)   # esqueleto: mínimo -> indicador
    dNew, gN = polegar(Gv, W_, ax)
    ang = math.atan2(dOld[1], dOld[0]) - math.atan2(dNew[1], dNew[0]); ang = (ang + math.pi) % (2 * math.pi) - math.pi
    print(S, 'LADO DO POLEGAR (y,z) esqueleto', dOld.round(2), 'malha', dNew.round(2), '| sobra', round(gN * 1000, 1), 'mm | giro do antebraço', round(math.degrees(ang), 1), 'graus')
    sel = s * (Gv[:, 0] - E_[0]) > 0; t = np.clip((Gv[sel] - E_) @ ax / np.linalg.norm(W_ - E_), 0, 1); w = t * t * (3 - 2 * t)
    for i, (idx, wi) in enumerate(zip(np.where(sel)[0], w)):
        Gv[idx] = E_ + np.array(mathutils.Matrix.Rotation(ang * wi, 3, mathutils.Vector(ax * s)) @ mathutils.Vector(Gv[idx] - E_))
    me.vertices.foreach_set('co', Gv.ravel()); me.update()
    # mão maior (MAO, padrão 1,15; a do Meshy tem o tamanho anatômico e parecia pequena perto da antiga): escala em torno
    # do pulso, entrando aos poucos entre 2 cm antes e 2 cm depois dele
    km = float(os.environ.get('MAO', 1.15)); uu = (Gv - W_) @ ax; t = np.clip((uu + 0.02) / 0.04, 0, 1); fe = 1 + (km - 1) * t * t * (3 - 2 * t)
    Gv[sel] = W_ + (Gv[sel] - W_) * fe[sel, None]; me.vertices.foreach_set('co', Gv.ravel()); me.update()
    # dedos pela própria malha (mão já de palma para baixo): além das membranas cada dedo é um pedaço solto da pele. Sobe um
    # corte (u = distância do pulso ao longo do braço) até sobrarem 4 pedaços compridos (o polegar, mais curto, fica de fora);
    # a membrana entre dois vizinhos é o corte mais baixo em que eles se separam; o nó fica 0,1 x mão antes das membranas
    # vizinhas. De cada nó, a linha do centro é seguida pela curva do dedo, presa ao seu pedaço de pele, até a ponta de
    # verdade (o mínimo vem bem dobrado para baixo); juntas 2, 3 e 4 ao longo dessa curva nas proporções do esqueleto antigo
    wr = J[S + 'Hand']; hm = s * (Gv[:, 0] - wr[0]) > 0; H = Gv[hm]; u = s * (H[:, 0] - wr[0]); L = u.max()
    uh = s * (Gv[:, 0] - wr[0]); Eh = ED[hm[ED[:, 0]] & hm[ED[:, 1]]]
    def rotulos(uc):     # pedaços soltos da pele acima do corte uc (rótulo = menor índice de vértice do pedaço)
        m = hm & (uh > uc); e = Eh[m[Eh[:, 0]] & m[Eh[:, 1]]]; lab = np.arange(len(Gv))
        while True:
            l2 = lab.copy(); mn = np.minimum(lab[e[:, 0]], lab[e[:, 1]]); np.minimum.at(l2, e[:, 0], mn); np.minimum.at(l2, e[:, 1], mn); l2 = l2[l2]
            if (l2 == lab).all(): return lab
            lab = l2
    for uc in np.arange(0.45, 0.9, 0.01) * L:
        lab = rotulos(uc); m = hm & (uh > uc); ids, cnt = np.unique(lab[m], return_counts=True)
        cp = [np.where(m & (lab == i_))[0] for i_, c_ in zip(ids, cnt) if c_ >= 15]; cp = [c_ for c_ in cp if uh[c_].max() > 0.65 * L]
        if len(cp) == 4: break
    cp.sort(key=lambda c_: Gv[c_, 1].mean())                                     # indicador primeiro (lado do polegar, -y)
    pon = [c_[np.argmax(np.linalg.norm(Gv[c_] - Gv[c_[uh[c_] < uh[c_].min() + 0.01]].mean(0), axis=1))] for c_ in cp]   # ponta: o mais longe da base do pedaço
    web = [None] * 3
    for uw in np.arange(0.3, 0.9, 0.01) * L:
        lab = rotulos(uw)
        for q in range(3):
            if web[q] is None and lab[pon[q]] != lab[pon[q + 1]]: web[q] = uw
        if None not in web: break
    sep = [web[0], max(web[0], web[1]), max(web[1], web[2]), web[2]]; nos = [web[0], (web[0] + web[1]) / 2, (web[1] + web[2]) / 2, web[2]]
    pedaco, yn = [], []
    for i_ in range(4):              # pedaço de cada dedo logo acima de onde ele se solta dos vizinhos; y do nó = base dele
        lab = rotulos(sep[i_] + 0.01 * L); c_ = np.where(lab == lab[pon[i_]])[0]; pedaco.append(c_)
        B_ = Gv[c_[uh[c_] < uh[c_].min() + 0.02 * L]]; yn.append((B_[:, 1].min() + B_[:, 1].max()) / 2)
    hw = 0.45 * np.diff(yn).min()
    print('   MEMBRANAS u', (np.array(web) / L).round(2), '(x mão) | corte dos 4 dedos', round(uc / L, 2), '| nós y', np.round(yn, 3), '| mão', round(L, 3), '| janela', round(hw * 1000, 1), 'mm')
    def secao(c0, n, keep, mv=None):   # contorno exato da mão no plano (c0, normal n): arestas que atravessam o plano; keep escolhe o dedo
        a, b = (Gv[ED[:, 0]] - c0) @ n, (Gv[ED[:, 1]] - c0) @ n; m = (a * b < 0) & hm[ED[:, 0]] & (True if mv is None else mv[ED[:, 0]]); e = ED[m]
        P = Gv[e[:, 0]] + (a[m] / (a[m] - b[m]))[:, None] * (Gv[e[:, 1]] - Gv[e[:, 0]]); return P[keep(P)]
    def meio(p, hw):
        P = secao(p, np.array([1.0, 0, 0]), lambda P: np.abs(P[:, 1] - p[1]) < hw); p = p.copy()
        if len(P) > 3: p[1:] = (P[:, 1:].min(0) + P[:, 1:].max(0)) / 2
        return p
    def prop(f):   # comprimento e frações das falanges no esqueleto antigo (média dos dois lados, que diferem)
        r = []
        for X in ('Left', 'Right'):
            o = [JO[X + 'Hand' + f + str(q)] for q in (1, 2, 3, 4)]; r.append([np.linalg.norm(o[q + 1] - o[q]) for q in range(3)])
        r = np.mean(r, 0); return r.sum(), r / r.sum()
    def linha(c_):   # linha do centro do pedaço do dedo: distância pela pele (arestas) a partir da base do pedaço; cada faixa
        pos = np.full(len(Gv), -1); pos[c_] = np.arange(len(c_))   # de distância é um anel em volta do dedo, mesmo onde ele dobra
        e_ = Eh[(pos[Eh[:, 0]] >= 0) & (pos[Eh[:, 1]] >= 0)]; viz = [[] for _ in c_]
        for a_, b_, w_ in zip(pos[e_[:, 0]], pos[e_[:, 1]], np.linalg.norm(Gv[e_[:, 0]] - Gv[e_[:, 1]], axis=1)): viz[a_].append((b_, w_)); viz[b_].append((a_, w_))
        dist = np.full(len(c_), np.inf); fila = [(0.0, k_) for k_ in np.where(uh[c_] < uh[c_].min() + 0.004)[0]]
        for _, k_ in fila: dist[k_] = 0
        while fila:
            d_, k_ = heapq.heappop(fila)
            if d_ > dist[k_]: continue
            for v_, w_ in viz[k_]:
                if d_ + w_ < dist[v_]: dist[v_] = d_ + w_; heapq.heappush(fila, (d_ + w_, v_))
        ok = np.isfinite(dist); fx = (dist / 0.004).astype(int)
        qs = [q for q in range(fx[ok].max() + 1) if (ok & (fx == q)).sum() >= 3]; C = np.array([Gv[c_[ok & (fx == q)]].mean(0) for q in qs])
        if len(C) > 2: C[1:-1] = (C[:-2] + C[1:-1] + C[2:]) / 3
        return C, c_[np.argmax(np.where(ok, dist, -1))], c_[ok], dist[ok], (np.array(qs) + 0.5) * 0.004   # + vértices, distâncias, centro das faixas
    for i_, f in enumerate(('Index', 'Middle', 'Ring', 'Pinky')):
        _, fr = prop(f); ub = nos[i_] - 0.1 * L; base = meio(np.array([wr[0] + s * ub, yn[i_], wr[2]]), hw)
        Cr, pt, cv, dv, dc = linha(pedaco[i_]); Cp = np.vstack([base, Cr, Gv[pt]]); sa = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Cp, axis=0), axis=1))]
        em = lambda t_: np.array([np.interp(t_, sa, Cp[:, c_]) for c_ in range(3)]); tot = sa[-1] - 0.002
        J[S + 'Hand' + f + '1'], J[S + 'Hand' + f + '2'], J[S + 'Hand' + f + '3'], J[S + 'Hand' + f + '4'] = base, em(fr[0] * tot), em((fr[0] + fr[1]) * tot), em(tot)
        sC = sa[1:1 + len(Cr)]; sv = np.interp(dv, dc, sC) + np.clip(dv - dc[-1], 0, None) + np.clip(dv - dc[0], None, 0)   # posição de cada vértice
        CL[S + f] = (Cp[:-1], np.array([0, fr[0] * tot, (fr[0] + fr[1]) * tot, tot]), cv, sv)                             # ao longo da linha
        da, db = Cp[2] - Cp[0], Cp[-1] - Cp[-3]
        print(f'   {S} {f:6s} nó u {ub:.3f} y {yn[i_]:+.3f} | ao longo do dedo {tot:.3f} (em linha reta {np.linalg.norm(Cp[-1] - base):.3f}, até a ponta da malha {np.linalg.norm(Gv[pon[i_]] - base):.3f}) | curvatura {np.degrees(np.arccos(da @ db / np.linalg.norm(da) / np.linalg.norm(db))):3.0f}°')
    # polegar: ponta = ponto mais à frente antes das pontas dos dedos; linha do centro rastreada da ponta para trás
    # (cortes perpendiculares, cada um centrado no anterior); juntas nas proporções do polegar antigo, total 0,48 x mão
    o = [JO[S + 'HandThumb' + str(q)] for q in (1, 2, 3, 4)]; d = (o[3] - o[0]) / np.linalg.norm(o[3] - o[0])
    side = H[u < 0.75 * L]; tt = side[np.argmax(side[:, 1:] @ dOld)]; c = tt.copy(); cs = []
    for q in np.arange(0.004, 0.25 * L, 0.003):
        c = c - d * 0.003; P = secao(c, d, lambda P: np.linalg.norm(P - c, axis=1) < 0.013 * km)
        if len(P) < 4: continue
        c = (P.min(0) + P.max(0)) / 2; cs.append(c)
        if len(cs) >= 4: C = np.array(cs); d2 = np.linalg.svd(C - C.mean(0))[2][0]; d = -d2 * np.sign(d2 @ (C[-1] - C[0]))
    C = np.array(cs); cc = C.mean(0); a0 = cc + d * ((tt - cc) @ d); Lt = 0.48 * L; _, fr = prop('Thumb')
    J[S + 'HandThumb4'] = a0 - d * 0.002
    for q, dist in ((3, fr[2] * Lt), (2, (fr[1] + fr[2]) * Lt), (1, Lt)): J[S + 'HandThumb' + str(q)] = a0 - d * dist
    J[S + 'HandThumb1'][2] = meio(J[S + 'HandThumb1'], 0.012)[2]                 # base do polegar no meio da espessura da mão
    print(f'   {S} Thumb  ponta {tt.round(3)} direção {d.round(2)} | {len(cs)} cortes | proporção antiga', fr.round(2), '| mão', round(L, 3))
# --- pernas e pés ---
for s, S in ((1, 'Left'), (-1, 'Right')):
    lm = (s * Gv[:, 0] > 0) & (Gv[:, 2] < crN); lmo = (s * Ov[:, 0] > 0) & (Ov[:, 2] < crO)
    for n in ('Leg', 'Foot'):
        zo = JO[S + n][2]; zn = zo                      # joelho e tornozelo: mesma fração da altura
        co, _ = slab(Ov, 2, zo, lmo); cn, P = slab(Gv, 2, zn, lm)
        J[S + n] = np.array([cn[0], cn[1] + (JO[S + n][1] - co[1]) * 0.5, zn])
    zo = JO[S + 'UpLeg'][2]; zn = UPZ
    hwO = np.abs(slab(Ov, 2, zo, Ov[:, 2] > 0)[1][:, 0]).max(); hwN = np.abs(slab(Gv, 2, zn, Gv[:, 2] > 0)[1][:, 0]).max()
    a, b = torso_y(Ov, zo); f = (JO[S + 'UpLeg'][1] - a) / (b - a); a2, b2 = torso_y(Gv, zn)
    J[S + 'UpLeg'] = np.array([JO[S + 'UpLeg'][0] * hwN / hwO, a2 + f * (b2 - a2), zn])
    fO = Ov[lmo & (Ov[:, 2] < JO[S + 'Foot'][2])]; fN = Gv[lm & (Gv[:, 2] < J[S + 'Foot'][2])]
    lo, hi, lo2, hi2 = fO.min(0), fO.max(0), fN.min(0), fN.max(0)
    for n in ('ToeBase', 'Toe_End'): J[S + n] = lo2 + (JO[S + n] - lo) / (hi - lo) * (hi2 - lo2)
    dO = JO[S + 'Leg'] - JO[S + 'UpLeg']; dN = J[S + 'Leg'] - J[S + 'UpLeg']
    print(S, 'PERNA abertura antiga', round(math.degrees(math.atan2(abs(dO[0]), -dO[2])), 1), 'nova', round(math.degrees(math.atan2(abs(dN[0]), -dN[2])), 1), 'graus')
J = {n: v for n, v in J.items() if not n.startswith('_')}
falta = [n for n in JO if n not in J]; print('JUNTAS', len(J), 'faltando', falta)
for n in ('Hips', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandMiddle1', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'):
    print(f'   {n:16s} antigo {JO[n].round(3)}  novo {J[n].round(3)}')

json.dump({n: list(map(float, v)) for n, v in J.items()}, open(os.path.join(OUT, 'juntas.json'), 'w'))
np.save(os.path.join(OUT, 'Gv_torcido.npy'), Gv)
if os.environ.get('DRY'): sys.exit(0)
# --- esqueleto auxiliar para o bone heat (ossos ao longo dos membros) ---
kids = {}
for n, i in NODES.items():
    p = par.get(i)
    if p is not None and jO['nodes'][p].get('name', '').startswith('mixamorig'): kids.setdefault(nm(jO['nodes'][p]['name']), []).append(n)
PRI = {'Hips': 'Spine', 'Spine2': 'Neck', 'LeftHand': 'LeftHandMiddle1', 'RightHand': 'RightHandMiddle1'}
ad = bpy.data.armatures.new('heat'); A = bpy.data.objects.new('heat', ad); bpy.context.scene.collection.objects.link(A)
bpy.context.view_layer.objects.active = A; A.select_set(True); bpy.ops.object.mode_set(mode='EDIT')
eb = {}
for n in JN:
    b = ad.edit_bones.new('mixamorig:' + n); b.head = J[n]
    ch = PRI.get(n) or (kids.get(n, [None])[0])
    tail = J[ch] if ch in J else J[n] + (J[n] - J[nm(jO['nodes'][par[NODES[n]]]['name'])]) * 0.5
    if np.linalg.norm(tail - J[n]) < 1e-4: tail = J[n] + np.array([0, 0, 0.01])
    b.tail = tail; eb[n] = b
for n in JN:
    p = par.get(NODES[n]); pn = nm(jO['nodes'][p]['name']) if p is not None else None
    if pn in eb: eb[n].parent = eb[pn]
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT'); G.select_set(True); A.select_set(True); bpy.context.view_layer.objects.active = A
bpy.ops.object.parent_set(type='ARMATURE_AUTO')
vg = {g.index: nm(g.name) for g in G.vertex_groups}; nV = len(me.vertices)
W = np.zeros((nV, len(JN)), np.float32); jix = {n: i for i, n in enumerate(JN)}
for v in me.vertices:
    for g in v.groups:
        if g.weight > 0 and vg[g.group] in jix: W[v.index, jix[vg[g.group]]] = g.weight
sem = W.sum(1) < 1e-4; print('PESOS: vértices sem peso do bone heat', int(sem.sum()), 'de', nV)
if sem.any():   # sem solução no heat: osso mais próximo (distância ao segmento)
    H = np.array([J[n] for n in JN]); Tl = np.array([np.array(A.data.bones['mixamorig:' + n].tail_local) for n in JN])
    P = Gv[sem]; ab = Tl - H; t = np.clip(((P[:, None] - H[None]) * ab[None]).sum(2) / np.maximum((ab * ab).sum(1), 1e-12)[None], 0, 1)
    d = np.linalg.norm(P[:, None] - (H[None] + t[..., None] * ab[None]), axis=2); W[np.where(sem)[0], d.argmin(1)] = 1
# lateral do tronco abaixo da axila: o bone heat deixa ali um pouco do braço e da clavícula (e da coxa nas costelas de
# baixo, e da cintura: 45% contra 1% no antigo); com o braço abaixado ou a perna erguida essa pele ia junto e afundava. Tira esses pesos do
# tronco, com transição suave, e o resto (coluna) assume
for s, S in ((1, 'Left'), (-1, 'Right')):
    A_ = J[S + 'Arm']; x_ = s * Gv[:, 0]; z_ = Gv[:, 2]; fx = np.clip((s * A_[0] - x_) / 0.03, 0, 1) * (x_ > 0.02)
    for n, z0, rz in ((S + 'Arm', A_[2] - 0.03, 0.06), (S + 'ForeArm', A_[2] - 0.03, 0.06), (S + 'Shoulder', A_[2] - 0.06, 0.08)):
        f_ = fx * np.clip((z0 - z_) / rz, 0, 1); print(f'   PESO {n:14s} tirado do tronco: {float((W[:, jix[n]] * f_).sum()):7.1f} (vértices {int(((W[:, jix[n]] > 0.01) & (f_ > 0)).sum())})'); W[:, jix[n]] *= 1 - f_
    U_ = J[S + 'UpLeg']; f_ = np.clip((z_ - U_[2] + 0.01) / 0.07, 0, 1)   # cintura e costelas: nada da coxa acima da articulação do quadril (como no modelo antigo)
    print(f'   PESO {S + "UpLeg":14s} tirado da cintura: {float((W[:, jix[S + "UpLeg"]] * f_).sum()):7.1f}'); W[:, jix[S + 'UpLeg']] *= 1 - f_
sem = W.sum(1) < 1e-4; W[sem, jix['Spine1']] = 1
top = np.argsort(-W, 1)[:, :4]; tw = np.take_along_axis(W, top, 1); tw /= tw.sum(1, keepdims=True)

# --- dedos na direção e no giro dos ossos: no esqueleto antigo os dedos ficam retos e juntos; os do Meshy vêm abertos em
# leque e dobrados. A animação dobra cada falange em torno de um eixo do osso antigo, então um dedo fora dessa direção, ou
# girado em torno de si mesmo, dobra de lado. Cada falange recebe a rotação que leva a sua direção para a do osso e o
# eixo em que o dedo do Meshy dobra para o eixo em que a animação dobra o osso (no polegar, só a direção); a malha é
# gravada nessa pose (skinning linear com os pesos acima) como o novo repouso e as juntas dos dedos vão junto.
def giro(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K / (1 + c)
BAIXO = np.array([0, 0, -1.0])                # lado da palma (mão de palma para baixo)
def eixo_anim(n, e1):   # eixo mais usado pelas animações para girar o osso n, no repouso (Blender); positivo = fechar o dedo
    k_ = NODES[n]; R0 = qm(jO['nodes'][k_].get('rotation', [0, 0, 0, 1])); rv = []
    for an in jO.get('animations', []):
        for ch in an['channels']:
            if ch['target'].get('node') != k_ or ch['target']['path'] != 'rotation': continue
            for q in acc(jO, bO, an['samplers'][ch['sampler']]['output']).astype(np.float64):
                Dm = R0.T @ qm(q / np.linalg.norm(q)); a = np.arccos(np.clip((np.trace(Dm) - 1) / 2, -1, 1))
                if a > 0.05: rv.append(np.array([Dm[2, 1] - Dm[1, 2], Dm[0, 2] - Dm[2, 0], Dm[1, 0] - Dm[0, 1]]) / (2 * np.sin(a)) * a)
    a = G2B(worldM(k_)[:3, :3] @ np.linalg.svd(np.array(rv), full_matrices=False)[2][0]); a /= np.linalg.norm(a)
    return a if np.cross(a, e1) @ BAIXO > 0 else -a
def base3(e1, e2):      # base ortonormal com e1 e a parte de e2 perpendicular a e1
    e1 = e1 / np.linalg.norm(e1); e2 = e2 - (e2 @ e1) * e1; e2 = e2 / np.linalg.norm(e2); return np.c_[e1, e2, np.cross(e1, e2)]
T = {}                                        # osso -> (pivô no repouso atual, pivô na pose nova, rotação)
for S in ('Left', 'Right'):
    for f in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
        n = [S + 'Hand' + f + str(q) for q in (1, 2, 3, 4)]; P0 = [J[x].copy() for x in n]; Pn = [P0[0]]; Gr = np.eye(3)
        if f != 'Thumb':                      # eixo em que o dedo do Meshy dobra (pela curva dele; reto: fechar para a palma)
            dA, dB = (P0[1] - P0[0]) / np.linalg.norm(P0[1] - P0[0]), (P0[3] - P0[2]) / np.linalg.norm(P0[3] - P0[2])
            fb = np.cross(dA, BAIXO); nf = np.cross(dA, dB) + 0.25 * fb / np.linalg.norm(fb)
        for q in range(3):
            d0, e1 = P0[q + 1] - P0[q], JO[n[q + 1]] - JO[n[q]]
            if f == 'Thumb': Gr = giro(Gr @ d0, e1) @ Gr; tw_ = ''
            else:
                a = eixo_anim(n[q], e1); Gr = base3(e1, a) @ base3(d0, nf).T
                v = giro(d0, e1) @ nf; v -= (v @ e1) * e1 / (e1 @ e1); ap = a - (a @ e1) * e1 / (e1 @ e1)
                tw_ = f' | giro em torno do dedo {np.degrees(np.arccos(np.clip(v @ ap / np.linalg.norm(v) / np.linalg.norm(ap), -1, 1))):4.1f}'
            T[n[q]] = (P0[q], Pn[q], Gr); Pn.append(Pn[q] + Gr @ d0)
            print(f'   {n[q]:16s} rotação {math.degrees(math.acos(np.clip(np.trace(Gr) / 2 - 0.5, -1, 1))):5.1f} graus{tw_}')
        T[n[3]] = (P0[3], Pn[3], Gr)
        for x, p in zip(n, Pn): J[x] = p
# Nos 4 dedos a pele não gira falange por falange (as emendas ficavam em zigue-zague): o dedo é desenrolado ao longo da
# linha do centro. Cada ponto da pele vai para a mesma distância ao longo do dedo já reto, com o mesmo afastamento da
# linha, girado pela rotação da falange (corrigida pela tangente local da curva e misturada perto das juntas).
def desenrola(S, f, idx, b=0.004):
    Cq, sJ, cv, sv = CL[S + f]; n = [S + 'Hand' + f + str(q) for q in (1, 2, 3, 4)]
    sq = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Cq, axis=0), axis=1))]
    t0 = (Cq[1] - Cq[0]) / np.linalg.norm(Cq[1] - Cq[0]); t1 = (Cq[-1] - Cq[-3]) / np.linalg.norm(Cq[-1] - Cq[-3])
    def ponto(s_):   # ponto da linha do centro na distância s_ (reta antes do nó e depois da última faixa)
        p = np.array([np.interp(s_, sq, Cq[:, c_]) for c_ in range(3)]).T
        return p + np.clip(s_, None, 0)[:, None] * t0 + np.clip(s_ - sq[-1], 0, None)[:, None] * t1
    X = Gv[idx]; Cx = np.vstack([Cq[0] - t0 * 0.04, Cq]); sa = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Cx, axis=0), axis=1))] - 0.04
    A_, AB = Cx[:-1], np.diff(Cx, axis=0); lAB = np.linalg.norm(AB, axis=1)   # fora do pedaço do dedo (palma): ponto mais perto
    t = np.clip(((X[:, None] - A_[None]) * AB[None]).sum(2) / (lAB ** 2)[None], 0, 1)
    k = np.linalg.norm(X[:, None] - (A_[None] + t[..., None] * AB[None]), axis=2).argmin(1); sp = sa[k] + t[np.arange(len(X)), k] * lAB[k]
    lk = np.full(len(Gv), np.nan); lk[cv] = sv; h_ = ~np.isnan(lk[idx]); sp[h_] = lk[idx][h_]      # no dedo: distância pela pele
    c = ponto(sp); tg = ponto(sp + 0.004) - ponto(sp - 0.004); tg /= np.linalg.norm(tg, axis=1)[:, None]
    r0 = X - c; ra = (r0 * tg).sum(1)[:, None] * tg; kg = 1 + (float(os.environ.get('DEDO', 1.2)) - 1) * np.clip(sp / 0.025, 0, 1)[:, None]
    X = c + ra + kg * (r0 - ra)                       # dedo mais grosso (DEDO x a distância à linha do centro), a partir do nó
    Pn = np.array([J[x] for x in n]); Rs = [T[x][2] for x in n[:3]]; e1 = [(Pn[q + 1] - Pn[q]) / np.linalg.norm(Pn[q + 1] - Pn[q]) for q in range(3)]
    q = np.clip(np.searchsorted(sJ, sp, 'right') - 1, 0, 2)
    ct = Pn[q] + ((sp - sJ[q]) / (sJ[q + 1] - sJ[q]))[:, None] * (Pn[q + 1] - Pn[q])
    out = np.empty_like(X)
    for i in range(len(X)):
        w = np.array([min(1 if j == 0 else (sp[i] - sJ[j] + b) / (2 * b), 1 if j == 2 else (sJ[j + 1] + b - sp[i]) / (2 * b)) for j in range(3)]).clip(0, 1)
        R = sum(w[j] * giro(Rs[j] @ tg[i], e1[j]) @ Rs[j] for j in range(3) if w[j] > 0) / w.sum()
        U_, _, Vt_ = np.linalg.svd(R); out[i] = ct[i] + (U_ @ Vt_) @ (X[i] - c[i])
    return out
# pontas dos 4 dedos com o mesmo espaço entre elas (o Meshy tem os nós desiguais: o anelar longe do médio e colado no
# mínimo): médio e anelar giram um pouco no plano da palma em torno do nó, juntas 2, 3 e 4 vão junto
for S in ('Left', 'Right'):
    fs = ('Index', 'Middle', 'Ring', 'Pinky'); yp = np.array([J[S + 'Hand' + f + '4'][1] for f in fs]); alvo = np.linspace(yp[0], yp[3], 4); ms = []
    for f, y0_, y1_ in zip(fs, yp, alvo):
        sJ = CL[S + f][1]
        for q in (2, 3, 4): J[S + 'Hand' + f + str(q)] = J[S + 'Hand' + f + str(q)] + np.array([0, (y1_ - y0_) * sJ[q - 1] / sJ[3], 0])
        ms.append(f'{f} {(y1_ - y0_) * 1000:+.1f} mm ({math.degrees(math.atan2(y1_ - y0_, sJ[3])):+.1f}°)')
    print('PONTAS IGUALADAS', S, ' | '.join(ms))
MP = {}
for S in ('Left', 'Right'):
    for f in ('Index', 'Middle', 'Ring', 'Pinky'):
        n = [S + 'Hand' + f + str(q) for q in (1, 2, 3, 4)]; bs = [jix[x] for x in n if x in jix]
        idx = np.where(np.isin(top, bs).any(1))[0]; Pm = Gv.copy(); Pm[idx] = desenrola(S, f, idx)
        for x in n: MP[x] = Pm
Gn = np.zeros_like(Gv); JNa = np.array(JN)
for c in range(4):
    for bi in np.unique(top[:, c]):
        m = top[:, c] == bi; nb = JNa[bi]; t = T.get(nb)
        Gn[m] += tw[m, c, None] * (MP[nb][m] if nb in MP else Gv[m] if t is None else t[1] + (Gv[m] - t[0]) @ t[2].T)
print('DEDOS: vértices movidos', int((np.linalg.norm(Gn - Gv, axis=1) > 1e-5).sum()), '| maior deslocamento', round(float(np.linalg.norm(Gn - Gv, axis=1).max()) * 100, 1), 'cm')
for S in ('Left', 'Right'):     # dedo pronto: centro de cada faixa da pele contra os ossos (mm)
    r__ = []
    for f in ('Index', 'Middle', 'Ring', 'Pinky'):
        _, sJ, cv, sv = CL[S + f]; a_, b_ = J[S + 'Hand' + f + '1'], J[S + 'Hand' + f + '4']; u_ = (b_ - a_) / np.linalg.norm(b_ - a_)
        Cn = np.array([Gn[cv[(sv >= x) & (sv < x + 0.006)]].mean(0) for x in np.arange(sv.min(), sv.max() - 0.006, 0.006) if ((sv >= x) & (sv < x + 0.006)).sum() >= 3])
        Pn_ = np.array([J[S + 'Hand' + f + str(q)] for q in (1, 2, 3, 4)]); A_, AB = Pn_[:-1], np.diff(Pn_, axis=0)
        t_ = np.clip(((Cn[:, None] - A_[None]) * AB[None]).sum(2) / (AB * AB).sum(1)[None], 0, 1); dq = Cn[:, None] - (A_[None] + t_[..., None] * AB[None])
        k_ = np.linalg.norm(dq, axis=2).argmin(1); dq = dq[np.arange(len(Cn)), k_]
        if os.environ.get('PERFIL'): print('   PERFIL', S, f, 'sJ', (sJ * 1000).round(0), '| s/lado/vert mm', [(round(x * 1000), round(dq[i, 1] * 1000, 1), round(dq[i, 2] * 1000, 1)) for i, x in enumerate(np.arange(sv.min(), sv.max() - 0.006, 0.006)[:len(Cn)])])
        r__.append(f'{f} {np.abs(dq[:, 1]).max() * 1000:4.1f} lado / {np.abs(dq[:, 2]).max() * 1000:4.1f} vertical (ossos {np.degrees(np.arccos(np.clip((AB[0] @ AB[2]) / np.linalg.norm(AB[0]) / np.linalg.norm(AB[2]), -1, 1))):3.0f}° de curva)')
    print('RETIDÃO', S, ' | '.join(r__))
Gv = Gn; me.vertices.foreach_set('co', Gv.ravel()); me.update()

# --- malha para o glTF: vértices separados por costura de UV, atlas (corpo à esquerda, cabeça à direita) ---
pz = [np.mean([Gv[list(p.vertices), 2].mean() for p in me.polygons if p.material_index == i]) for i in range(len(me.materials))]
iHead = int(np.argmax(pz)); print('MATERIAIS', [m.name for m in me.materials], 'cabeça =', me.materials[iHead].name)
me.calc_loop_triangles(); nt = len(me.loop_triangles)
lt = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get('loops', lt)
tp = np.empty(nt, np.int32); me.loop_triangles.foreach_get('polygon_index', tp)
pm = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get('material_index', pm)
uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', lv)
print('UV faixa', uv.min(0).round(3), uv.max(0).round(3))
lm_ = np.repeat(pm[tp], 3); L = lt
key = np.stack([lv[L], np.round(uv[L, 0] * 1e5).astype(np.int64), np.round(uv[L, 1] * 1e5).astype(np.int64), lm_ == iHead], 1)
uk, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
first = np.zeros(len(uk), np.int64); first[inv] = np.arange(len(L))
vi = lv[L][first]; uvo = uv[L][first].astype(np.float64); hd = (lm_ == iHead)[first]
uvo[:, 0] = np.clip(uvo[:, 0], 0, 1) * 0.5 + 0.5 * hd; uvo[:, 1] = 1 - np.clip(uvo[:, 1], 0, 1)
vn = np.empty(nV * 3, np.float32); me.vertex_normals.foreach_get('vector', vn); vn = vn.reshape(-1, 3)
B2G = lambda p: np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)
JG = {n: B2G(v[None])[0] * k for n, v in J.items()}
np.savez(os.path.join(OUT, 'guerreiro_rig.npz'), P=(B2G(Gv[vi]) * k).astype(np.float32), N=B2G(vn[vi]).astype(np.float32), UV=uvo.astype(np.float32),
         IDX=inv.astype(np.uint32), JNT=np.array([[jix[JN[c]] for c in r] for r in top[vi]], np.uint16), WGT=tw[vi].astype(np.float32),
         JNAMES=np.array(list(JG)), JPOS=np.array([JG[n] for n in JG], np.float64))
print('MALHA glTF: vértices', len(vi), 'triângulos', nt)

# atlas 4096 x 2048
def base(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'TEX_IMAGE' and n.image and 'metal' not in n.image.name.lower(): return n.image
ims = [base(me.materials[i]) for i in range(len(me.materials))]; body = ims[1 - iHead]; head = ims[iHead]
At = np.zeros((2048, 4096, 4), np.float32); At[..., 3] = 1
for im, x0 in ((body, 0), (head, 2048)):
    im2 = im.copy(); im2.scale(2048, 2048); a = np.empty(2048 * 2048 * 4, np.float32); im2.pixels.foreach_get(a); At[:, x0:x0 + 2048] = a.reshape(2048, 2048, 4)
at = bpy.data.images.new('atlas', 4096, 2048); at.pixels.foreach_set(At.ravel())
sc = bpy.context.scene; sc.render.image_settings.file_format = 'JPEG'; sc.render.image_settings.quality = 88
sc.view_settings.view_transform = 'Standard'; at.save_render(os.path.join(OUT, 'atlas.jpg'), scene=sc)
print('ATLAS', os.path.getsize(os.path.join(OUT, 'atlas.jpg')) // 1024, 'KB')
