# junta cabeça + corpo masculinos (Meshy) com proporção anatômica:
#   cabeça = 13% da altura, queixo ~10 cm acima da base do pescoço; pescoço da cabeça esticado até o corpo,
#   base do pescoço do corpo alargada, emenda soldada e suavizada, cor da pele misturada na emenda
import bpy, bmesh, sys, os, numpy as np, mathutils, math
D = os.path.dirname(__file__); sys.path.insert(0, D); from rl import *; from common import coords, edges, components, E
CORPO = r'C:\Users\Filipe\Downloads\Meshy_AI_athletic_male_body_qu_1102030627_texture.glb'
CABECA = r'C:\Users\Filipe\Downloads\Meshy_AI_warrior_head_quad_gam_1102030637_texture.glb'
OUTD = os.environ.get('OUTD', r'C:\Users\Filipe\Downloads\guerreiro_mixamo')
S = E('S', 0.174)                      # escala da cabeça (altura queixo-topo = 13% da estatura)
MH = E('MH', -0.52)                    # z do mento (ponta de baixo do queixo) na cabeça original
DZ = E('MENTON', 0.851) - S * MH       # mento ~10 cm (5,2% da altura) acima da fúrcula do esterno
BAND, BANDB, K = E('BAND', 0.04), E('BANDB', 0.04), E('K', 0)
bpy.ops.wm.read_factory_settings(use_empty=True)

def imp(f, name):
    n = set(bpy.data.objects); bpy.ops.import_scene.gltf(filepath=f, merge_vertices=True)
    o = [o for o in set(bpy.data.objects) - n if o.type == 'MESH'][0]; o.name = name
    mw = o.matrix_world.copy(); o.parent = None; o.data.transform(mw); o.matrix_world.identity(); return o

def weld(o, dist):
    bm = bmesh.new(); bm.from_mesh(o.data); n0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=dist)
    nb = sum(1 for e in bm.edges if e.is_boundary); n1 = len(bm.verts); bm.to_mesh(o.data); bm.free()
    print('SOLDA GERAL', o.name, 'fundidos', n0 - n1, 'bordas abertas', nb)

def base_img(o):
    for n in o.active_material.node_tree.nodes:
        if n.type == 'TEX_IMAGE' and 'metallic' not in n.image.name: return n.image

def vcolor(o):
    me = o.data; img = base_img(o); W, H = img.size
    a = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(a); tex = a.reshape(H, W, 4)
    uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', lv)
    tu = np.clip((uv[:, 0] % 1) * (W - 1), 0, W - 1).astype(int); tv = np.clip((uv[:, 1] % 1) * (H - 1), 0, H - 1).astype(int)
    vc = np.zeros((len(me.vertices), 3)); cnt = np.zeros(len(me.vertices))
    np.add.at(vc, lv, tex[tv, tu, :3]); np.add.at(cnt, lv, 1); return vc / np.maximum(cnt, 1)[:, None]

def cut(o, keep_below, region=None):
    bm = bmesh.new(); bm.from_mesh(o.data)
    fs = [f for f in bm.faces if region is None or region(f.calc_center_median())]
    geom = fs + list({e for f in fs for e in f.edges}) + list({v for f in fs for v in f.verts})
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-7, plane_co=(0, 0, P), plane_no=(0, 0, 1),
                           clear_outer=keep_below and region is None, clear_inner=not keep_below)
    if keep_below and region is not None:
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > P + 1e-6 and region(v.co)], context='VERTS')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    ring = np.array([v.co[:] for v in bm.verts if abs(v.co.z - P) < 1e-5 and v.is_boundary])
    bm.to_mesh(o.data); bm.free(); return ring

body = imp(CORPO, 'corpo'); head = imp(CABECA, 'cabeca')
for o in list(bpy.data.objects):
    if o.type != 'MESH': bpy.data.objects.remove(o)
for o in (body, head): weld(o, E('WELD', 1e-4))
head.data.transform(mathutils.Matrix.Translation((0, 0, DZ)) @ mathutils.Matrix.Scale(S, 4))

# borda da "tigela" do pescoço do corpo: ponto mais alto da parede em cada direção; gira tudo p/ ela ficar horizontal
cbr = coords(body); rr = np.hypot(cbr[:, 0], cbr[:, 1]); tht = np.arctan2(cbr[:, 1], cbr[:, 0])
sel = (rr > 0.035) & (rr < 0.10) & (cbr[:, 2] > 0.76); pts = []
for k in range(48):
    m = sel & (((tht + np.pi) / (2 * np.pi) * 48).astype(int) % 48 == k)
    if m.any(): pts.append(cbr[m][np.argmax(cbr[m, 2])])
pts = np.array(pts); c0 = pts.mean(0); nrm = np.linalg.svd(pts - c0)[2][2]; nrm = nrm if nrm[2] > 0 else -nrm
Mrot = (mathutils.Matrix.Translation(mathutils.Vector(c0)) @ mathutils.Vector(nrm).rotation_difference((0, 0, 1)).to_matrix().to_4x4()
        @ mathutils.Matrix.Translation(-mathutils.Vector(c0)))
for o in (body, head): o.data.transform(Mrot)
res = (pts - c0) @ nrm; P = c0[2] + res.min() - E('OFF', 0.003)
print('BORDA inclinação', round(math.degrees(math.acos(nrm[2])), 1), 'graus | corte z', round(P, 4))

# estica o pescoço da cabeça para baixo (abaixo do queixo) até passar do corte em toda a volta
mz = (Mrot @ mathutils.Vector((0, -0.44 * S, DZ + MH * S))).z; Zs = mz - E('ZS', 0.008)
ch = coords(head); low = ch[:, 2] < Zs
d = ch[:, :2] - c0[:2]; th = np.arctan2(d[:, 1], d[:, 0]); bins = ((th + np.pi) / (2 * np.pi) * 36).astype(int) % 36
kk = []
for b in range(36):
    m = low & (bins == b)
    if m.sum() > 5: kk.append((Zs - (P - 0.012)) / max(Zs - ch[m, 2].min(), 1e-6))
kst = min(max(1.0, max(kk)), 1.8)
ch[low, 2] = Zs - (Zs - ch[low, 2]) * kst; head.data.vertices.foreach_set('co', ch.ravel()); head.data.update()
print('PESCOCO esticado x', round(kst, 3), 'a partir de z', round(Zs, 4), '(queixo em', round(mz, 4), ')')

# tom global: pele dos dois lados do corte
cb, ch = coords(body), coords(head); vb, vh = vcolor(body), vcolor(head)
sb = (cb[:, 2] > P - 0.03) & (cb[:, 2] < P + 0.02) & (np.abs(cb[:, 0] - c0[0]) < 0.1)
sh = (ch[:, 2] > P) & (ch[:, 2] < P + 0.04)
gain = np.median(vb[sb], 0) / np.median(vh[sh], 0)
print('TOM corpo', np.median(vb[sb], 0).round(3), 'cabeca', np.median(vh[sh], 0).round(3), 'ganho', gain.round(3))

neck = lambda c: abs(c.x - c0[0]) < 0.12 and abs(c.y - c0[1]) < 0.12 and c.z > P - 0.07
cut(body, True, neck)
cbn = coords(body); lab = components(len(cbn), edges(body), np.ones(len(cbn), bool))
far = (np.hypot(cbn[:, 0] - c0[0], cbn[:, 1] - c0[1]) > 0.1) | (cbn[:, 2] < P - 0.08)
junk = ~np.isin(lab, np.unique(lab[far]))
vn = np.empty(len(cbn) * 3, np.float32); body.data.vertex_normals.foreach_get('vector', vn); vn = vn.reshape(-1, 3)
rd = cbn[:, :2] - c0[:2]; rd /= np.maximum(np.linalg.norm(rd, axis=1), 1e-9)[:, None]; outn = (vn[:, :2] * rd).sum(1)
for k in np.unique(lab[junk]):
    m = lab == k
    if outn[m].mean() > 0.3: junk &= ~m          # pele da parede do pescoço (virada para fora): fica
print('tigela removida', junk.sum(), 'vértices')
bm = bmesh.new(); bm.from_mesh(body.data); bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.where(junk)[0]], context='VERTS')
ring_b = np.array([v.co[:] for v in bm.verts if abs(v.co.z - P) < 1e-5 and v.is_boundary])
bm.to_mesh(body.data); bm.free()
ring_h = cut(head, False)
off = np.r_[ring_b[:, :2].mean(0) - ring_h[:, :2].mean(0), 0] * E('CEN', 1)
head.data.transform(mathutils.Matrix.Translation(mathutils.Vector(off))); ring_h = ring_h + off
print('anel corpo', len(ring_b), 'cabeca', len(ring_h), 'desloc xy', off[:2].round(4))

cen = ring_b[:, :2].mean(0)
def radial(ring, n=72):
    d = ring[:, :2] - cen; th = np.arctan2(d[:, 1], d[:, 0]); r = np.hypot(d[:, 0], d[:, 1])
    b = ((th + np.pi) / (2 * np.pi) * n).astype(int) % n; R = np.full(n, np.nan)
    for k in range(n):
        s = r[b == k]
        if len(s): R[k] = np.median(s)
    i = np.where(~np.isnan(R))[0]
    return np.interp(np.arange(n), np.concatenate([i - n, i, i + n]), np.tile(R[i], 3))
def at(R, th):
    n = len(R); x = (th + np.pi) / (2 * np.pi) * n - 0.5
    return np.interp(x, np.arange(-1, n + 1), np.concatenate([[R[-1]], R, [R[0]]]))
Rb, Rh = radial(ring_b), radial(ring_h); Rt = Rb + K * (Rh - Rb)
print('raio corpo médio', Rb.mean().round(4), '| cabeca médio', Rh.mean().round(4), '| razão', (Rh.mean() / Rb.mean()).round(3))
def blend(o, R0, sgn, band_h):
    c = coords(o); t = sgn * (c[:, 2] - P) / band_h; d = c[:, :2] - cen; th = np.arctan2(d[:, 1], d[:, 0]); r = np.hypot(d[:, 0], d[:, 1])
    m = (t > -1e-6) & (t < 1)
    if sgn < 0: m &= r < at(R0, th) + 0.05
    tt = np.clip(t[m], 0, 1); w = (1 - tt) ** 2 * (1 + 2 * tt)
    if sgn < 0: w *= np.clip(1 - (r[m] - at(R0, th[m])) / 0.05, 0, 1)
    rn = r[m] + w * (at(Rt, th[m]) - at(R0, th[m]))
    c[m, :2] = cen + d[m] / np.maximum(r[m], 1e-9)[:, None] * rn[:, None]
    o.data.vertices.foreach_set('co', c.ravel()); o.data.update()
blend(head, Rh, 1, BAND); blend(body, Rb, -1, BANDB)

# textura da cabeça: tom global + correção por ângulo que some 5 cm acima da emenda
img = base_img(head); W, H = img.size
a = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(a); a = a.reshape(H, W, 4); a[..., :3] *= gain
NB = 24
def angcol(c, v, m):
    th = np.arctan2(c[m, 1] - cen[1], c[m, 0] - cen[0]); b = ((th + np.pi) / (2 * np.pi) * NB).astype(int) % NB
    C = np.full((NB, 3), np.nan)
    for k in range(NB):
        if (b == k).sum() > 2: C[k] = np.median(v[m][b == k], 0)
    i = np.where(~np.isnan(C[:, 0]))[0]
    return np.stack([np.interp(np.arange(NB), np.concatenate([i - NB, i, i + NB]), np.tile(C[i, j], 3)) for j in range(3)], 1)
cb, ch = coords(body), coords(head); vb, vh = vcolor(body), vcolor(head) * gain
rb_ = np.hypot(cb[:, 0] - cen[0], cb[:, 1] - cen[1])
Cb = angcol(cb, vb, (cb[:, 2] > P - 0.02) & (cb[:, 2] <= P + 1e-6) & (rb_ < 0.11))
Ch = angcol(ch, vh, (ch[:, 2] >= P - 1e-6) & (ch[:, 2] < P + 0.02))
F = np.clip(Cb / np.maximum(Ch, 1e-4), 0.6, 1.6)
F = (np.roll(F, 1, 0) + 2 * F + np.roll(F, -1, 0)) / 4
print('COR emenda: fator por ângulo min', F.min(0).round(3), 'max', F.max(0).round(3))
BT = E('BT', 0.05); me = head.data; me.calc_loop_triangles()
lt = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get('loops', lt); lt = lt.reshape(-1, 3)
uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', lv)
TP = ch[lv[lt]]; TU = uv[lt] * np.array([W, H])
sel_t = np.where(TP[:, :, 2].min(1) < P + BT)[0]
def rast(i):   # texels (linha, coluna) cobertos pelo triângulo i e a posição 3D de cada um
    p, q = TP[i], TU[i]
    x0, y0 = np.floor(q.min(0)).astype(int) - 1; x1, y1 = np.ceil(q.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
    if x1 < x0 or y1 < y0: return None
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    v0, v1 = q[1] - q[0], q[2] - q[0]; den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-9: return None
    rx, ry = gx - q[0, 0], gy - q[0, 1]
    l1 = (rx * v1[1] - v1[0] * ry) / den; l2 = (v0[0] * ry - rx * v0[1]) / den; l0 = 1 - l1 - l2
    yy, xx = np.where((l0 > -0.08) & (l1 > -0.08) & (l2 > -0.08))
    if not len(yy): return None
    wp = l0[yy, xx, None] * p[0] + l1[yy, xx, None] * p[1] + l2[yy, xx, None] * p[2]
    return y0 + yy, x0 + xx, wp
mult = np.ones((H, W, 3), np.float32); ntx = 0
for i in sel_t:
    rs = rast(i)
    if rs is None: continue
    yy, xx, wp = rs
    tt = np.clip((wp[:, 2] - P) / BT, 0, 1); w = (1 - tt) ** 2 * (1 + 2 * tt)
    thp = np.arctan2(wp[:, 1] - cen[1], wp[:, 0] - cen[0]); xb = (thp + np.pi) / (2 * np.pi) * NB - 0.5
    f = np.stack([np.interp(xb, np.arange(-1, NB + 1), np.concatenate([[F[-1, j]], F[:, j], [F[0, j]]])) for j in range(3)], -1)
    mult[yy, xx] = 1 + w[:, None] * (f - 1); ntx += len(yy)
a[..., :3] = np.clip(a[..., :3] * mult, 0, 1)
img.pixels.foreach_set(a.ravel()); img.update(); img.pack()
print('COR emenda: triângulos', len(sel_t), 'texels ajustados', ntx)

# junta, solda os anéis e suaviza a faixa da emenda
org = head.data.attributes.new('origem', 'INT', 'POINT'); org.data.foreach_set('value', np.ones(len(head.data.vertices), np.int32))
for o in (body, head): o.select_set(True)
with bpy.context.temp_override(active_object=body, object=body, selected_objects=[body, head], selected_editable_objects=[body, head]):
    bpy.ops.object.join()
me = body.data; bm = bmesh.new(); bm.from_mesh(me); lay = bm.verts.layers.int['origem']
def ring_edges(src):
    return [e for e in bm.edges if e.is_boundary and all(abs(v.co.z - P) < 1e-5 and v[lay] == src for v in e.verts)]
def snap(src_verts, dst_edges):
    for v in src_verts:
        best = None
        for e in dst_edges:
            a_, b_ = e.verts[0].co, e.verts[1].co; ab = b_ - a_
            f = max(0.0, min(1.0, (v.co - a_).dot(ab) / max(ab.length_squared, 1e-12))); p = a_ + ab * f
            dd = (v.co - p).length
            if best is None or dd < best[0]: best = (dd, e, f, p)
        dd, e, f, p = best
        if f * e.calc_length() < 1e-5: v.co = e.verts[0].co.copy(); continue
        if (1 - f) * e.calc_length() < 1e-5: v.co = e.verts[1].co.copy(); continue
        ne, nv = bmesh.utils.edge_split(e, e.verts[0], f); nv.co = p
        nv[lay] = e.verts[0][lay]; v.co = p.copy()
        dst_edges.remove(e); dst_edges += [ne, e]
snap([v for v in bm.verts if v.is_boundary and abs(v.co.z - P) < 1e-5 and v[lay] == 1], ring_edges(0))
snap([v for v in bm.verts if v.is_boundary and abs(v.co.z - P) < 1e-5 and v[lay] == 0], ring_edges(1))
n0 = len(bm.verts); bmesh.ops.remove_doubles(bm, verts=[v for v in bm.verts if abs(v.co.z - P) < 1e-5], dist=2e-5)
nb0 = sum(1 for e in bm.edges if e.is_boundary)
print('SOLDA emenda fundidos', n0 - len(bm.verts), '| bordas abertas no corte',
      sum(1 for e in bm.edges if e.is_boundary and all(abs(v.co.z - P) < 1e-5 for v in e.verts)))

# "escultura" do pescoço: perfil radial r(ângulo, altura) de baixa frequência; na faixa da emenda ele vira uma curva lisa
# (Hermite monotônica) que casa raio e inclinação do corpo embaixo e do pescoço da cabeça em cima — o detalhe fino da pele fica
ZA, ZB, DS, NT, DZG = P - E('SA', 0.03), P + min(E('SB', 0.055), mz - P - 0.03), 0.0125, 36, 0.0025
bm.verts.ensure_lookup_table(); C = np.array([v.co[:] for v in bm.verts])
dv = C[:, :2] - cen; rv = np.hypot(dv[:, 0], dv[:, 1]); tv = np.arctan2(dv[:, 1], dv[:, 0])
reg = (C[:, 2] > ZA - DS - 0.012) & (C[:, 2] < ZB + DS + 0.012) & (rv < 0.16)
tg = np.linspace(-np.pi, np.pi, NT, endpoint=False); zg = np.arange(ZA - DS, ZB + DS + 1e-9, DZG)
def perfil(rr, tt, zz):
    G = np.zeros((NT, len(zg)))
    for i, t0 in enumerate(tg):
        dt = np.angle(np.exp(1j * (tt - t0))); nr = np.abs(dt) < 0.6
        w = np.exp(-(dt[nr, None] * 0.07 / 0.012) ** 2 - ((zz[nr, None] - zg[None, :]) / 0.004) ** 2)
        G[i] = (w * rr[nr, None]).sum(0) / np.maximum(w.sum(0), 1e-12)
    return G
def alvo(G):
    Tg = G.copy(); L = ZB - ZA; m = (zg > ZA) & (zg < ZB); t = (zg[m] - ZA) / L
    for i in range(NT):
        f = lambda z: np.interp(z, zg, G[i])
        r0, r1 = f(ZA), f(ZB); dl = r1 - r0
        s0, s1 = (f(ZA) - f(ZA - DS)) / DS * L, (f(ZB + DS) - f(ZB)) / DS * L
        s0, s1 = [0.0 if s * dl <= 0 else np.sign(s) * min(abs(s), 3 * abs(dl)) for s in (s0, s1)]
        Tg[i, m] = (2*t**3 - 3*t**2 + 1) * r0 + (t**3 - 2*t**2 + t) * s0 + (-2*t**3 + 3*t**2) * r1 + (t**3 - t**2) * s1
    return Tg
def nome(t):
    return ['lado dir.', 'frente-dir.', 'frente', 'frente-esq.', 'lado esq.', 'nuca-esq.', 'nuca', 'nuca-dir.'][int(round((math.degrees(t) + 180) / 45)) % 8]
G = perfil(rv[reg], tv[reg], C[reg, 2]); Dg = alvo(G) - G
print('IMPERFEIÇÕES do perfil do pescoço (+ = buraco/afundado, - = calombo/degrau para fora):')
for i in np.argsort(-np.abs(Dg).max(1))[:8]:
    j = np.argmax(np.abs(Dg[i])); print(f'   {nome(tg[i] + np.pi / NT):11s} {math.degrees(tg[i]):6.0f}°  {Dg[i, j] * 1000:+6.1f} mm  a {(zg[j] - P) * 100:+5.1f} cm da emenda')
print('   desvio médio', round(np.abs(Dg).mean() * 1000, 2), 'mm | máximo', round(np.abs(Dg).max() * 1000, 2), 'mm')
band = (C[:, 2] > ZA) & (C[:, 2] < ZB) & (rv < 0.16)
xi = (tv[band] + np.pi) / (2 * np.pi) * NT; i0 = np.floor(xi).astype(int) % NT; i1 = (i0 + 1) % NT; fx = xi - np.floor(xi)
zi = (C[band, 2] - zg[0]) / DZG; j0 = np.clip(np.floor(zi).astype(int), 0, len(zg) - 2); fz = np.clip(zi - j0, 0, 1)
dd = (1 - fx) * (1 - fz) * Dg[i0, j0] + fx * (1 - fz) * Dg[i1, j0] + (1 - fx) * fz * Dg[i0, j0 + 1] + fx * fz * Dg[i1, j0 + 1]
C[band, :2] = cen + dv[band] / np.maximum(rv[band], 1e-9)[:, None] * (rv[band] + dd)[:, None]
for k in np.where(band)[0]: bm.verts[k].co = C[k]
print('ESCULPIDOS', band.sum(), 'vértices entre', round((ZA - P) * 100, 1), 'e +', round((ZB - P) * 100, 1), 'cm da emenda')
sv = [v for v in bm.verts if abs(v.co.z - P) < E('SMB', 0.015) and math.hypot(v.co.x - cen[0], v.co.y - cen[1]) < 0.11 and not v.is_boundary]
for _ in range(int(E('SMI', 6))):
    bmesh.ops.smooth_vert(bm, verts=sv, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
print('suavizados', len(sv), '| bordas abertas total', nb0, '->', sum(1 for e in bm.edges if e.is_boundary))
C2 = np.array([v.co[:] for v in bm.verts]); d2 = C2[:, :2] - cen; r2 = np.hypot(d2[:, 0], d2[:, 1])
reg2 = (C2[:, 2] > ZA - DS - 0.012) & (C2[:, 2] < ZB + DS + 0.012) & (r2 < 0.16)
G2 = perfil(r2[reg2], np.arctan2(d2[reg2, 1], d2[reg2, 0]), C2[reg2, 2]); D2 = alvo(G2) - G2
print('DEPOIS: desvio médio', round(np.abs(D2).mean() * 1000, 2), 'mm | máximo', round(np.abs(D2).max() * 1000, 2), 'mm')
bm.to_mesh(me); bm.free(); me.attributes.remove(me.attributes['origem'])
with bpy.context.temp_override(object=body, active_object=body):
    try: bpy.ops.mesh.customdata_custom_splitnormals_clear()
    except Exception as ex: print('normais', ex)
me.shade_smooth(); me.transform(Mrot.inverted())
cc = coords(body); me.transform(mathutils.Matrix.Translation((-(cc[:, 0].min() + cc[:, 0].max()) / 2, -(cc[:, 1].min() + cc[:, 1].max()) / 2, -cc[:, 2].min())))
cc = coords(body); body.name = 'guerreiro'
T = cc[:, 2].max(); print('FINAL verts', len(me.vertices), 'tris', sum(len(p.vertices) - 2 for p in me.polygons), 'altura', round(T, 3),
      '| cabeça/altura', round(S * (0.9507 - MH) / T, 3), '= 1 /', round(T / (S * (0.9507 - MH)), 2), 'cabeças')

os.makedirs(OUTD, exist_ok=True)
for im in bpy.data.images:
    if im.size[0] == 0: continue
    ext = '.png' if im.file_format == 'PNG' else '.jpg'
    nm = ('cabeca_' if im == img else '') + im.name.replace('.', '_') + ext
    im.filepath_raw = os.path.join(OUTD, nm); im.file_format = 'PNG' if ext == '.png' else 'JPEG'; im.save()
bpy.context.view_layer.objects.active = body; body.select_set(True)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUTD, 'guerreiro.glb'), export_format='GLB')
bpy.ops.export_scene.fbx(filepath=os.path.join(OUTD, 'guerreiro.fbx'), path_mode='COPY', embed_textures=True, object_types={'MESH'})
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTD, 'guerreiro.blend'))
b1 = os.path.join(OUTD, 'guerreiro.blend1')
if os.path.exists(b1): os.remove(b1)
for f in sorted(os.listdir(OUTD)): print('ARQ', f, round(os.path.getsize(os.path.join(OUTD, f)) / 1e6, 1), 'MB')

matte(body); co = setup_render((800, 800), 32)
TAG = os.environ.get('TAG', 'p'); hz = T - 0.12
for az in (0, 35, 90, 180): shoot(co, (0, 0, hz - 0.06), 0.9, az, 3, f'{D}/out2/{TAG}_busto{az}.png', lens=60)
hc = cc[cc[:, 2] > T - 0.25].mean(0); shoot(co, (hc[0] + 0.06, hc[1] + 0.02, T - 0.14), 0.4, 115, 5, f'{D}/out2/{TAG}_orelha.png', lens=60)
sc = bpy.context.scene; sc.render.resolution_x, sc.render.resolution_y = 700, 900
for az in (0,): shoot(co, (0, 0, T / 2), 3.6, az, 3, f'{D}/out2/{TAG}_inteiro{az}.png', lens=50)
