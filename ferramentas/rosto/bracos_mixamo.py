# uso: python3 ferramentas/rosto/bracos_mixamo.py (gera mix_tex_bracos.png) e depois ferramentas/trocar_textura.py
# projeta as vistas dorsal e palmar (arm_ref.webp) nos braços do personagem do Mixamo
import numpy as np, io
from PIL import Image, ImageFilter
from glb import load, acc
j,b=load('/home/user/aethelred-game/aethelred_personagem.glb')
pr=j['meshes'][0]['primitives'][0];at=pr['attributes']
P=acc(j,b,at['POSITION']).astype(float);N=acc(j,b,at['NORMAL']).astype(float);UV=acc(j,b,at['TEXCOORD_0']).astype(float)
I=acc(j,b,pr['indices']).ravel();Jn=acc(j,b,at['JOINTS_0']);W=acc(j,b,at['WEIGHTS_0']).astype(float)
if W.max()>1.5:W/=255
names=[j['nodes'][n]['name'].replace('mixamorig:','') for n in j['skins'][0]['joints']]
v=j['bufferViews'][j['images'][0]['bufferView']];o=v.get('byteOffset',0)
TEX=np.asarray(Image.open(io.BytesIO(b[o:o+v['byteLength']])).convert('RGB'),np.float32);TH,TW=TEX.shape[:2]
SRC=np.asarray(Image.open('/home/user/aethelred-game/rostos/referencia_bracos.webp').convert('RGB'),np.float32)
armw=np.zeros(len(P))
for k,n in enumerate(names):
    if any(t in n for t in ('Arm','Hand')) and 'Shoulder' not in n: armw+=((Jn==k)*W).sum(1)
X=np.abs(P[:,0])
LM=np.array([0.00211,0.00464,0.0069,0.00858,np.abs(P[:,0]).max()])   # ombro, cotovelo, pulso, base dos dedos, ponta
def s_of(x):return np.interp(x,LM,[0,1,2,3,4],left=-1+ (x-LM[0])/(LM[1]-LM[0]) if np.ndim(x)==0 else None)
def sx(x):
    s=np.interp(x,LM,[0,1,2,3,4]); lo=x<LM[0]; s=np.where(lo,(x-LM[0])/(LM[1]-LM[0]),s); return s
# vistas: [x da imagem em s=0..4], faixa de linhas
VIEWS={'dorso':dict(xs=[60,290,492,590,667],rows=(30,215)),'palma':dict(xs=[640,420,205,110,33],rows=(270,450))}
mx=SRC.max(2);mn=SRC.min(2);msk=(mx-mn)>22
def silh(V):
    r0,r1=V['rows'];m=msk[r0:r1];top=np.full(700,np.nan);bot=np.full(700,np.nan)
    for x in range(700):
        c=np.nonzero(m[:,x])[0]
        if len(c):top[x]=c.min()+r0;bot[x]=c.max()+r0
    # suaviza e preenche
    ok=~np.isnan(top);xs=np.arange(700);top=np.interp(xs,xs[ok],top[ok]);bot=np.interp(xs,xs[ok],bot[ok])
    k=np.ones(9)/9;top=np.convolve(top,k,'same');bot=np.convolve(bot,k,'same')
    return top,bot
for V in VIEWS.values():V['top'],V['bot']=silh(V)
# perfil do modelo: centro e meia-largura em z e centro em y por fatia de s
sel=(armw>0.5)&(X>0.0017)
sv=sx(X)
bins=np.linspace(-0.4,4,45);zc=np.zeros(len(bins));hz=np.zeros(len(bins));yc=np.zeros(len(bins))
for i,c in enumerate(bins):
    m=sel&(np.abs(sv-c)<0.06)
    if m.sum()<5:m=sel&(np.abs(sv-c)<0.15)
    if m.sum()<5:zc[i]=hz[i]=yc[i]=np.nan;continue
    z=P[m,2];y=P[m,1];zc[i]=(np.percentile(z,98)+np.percentile(z,2))/2;hz[i]=(np.percentile(z,98)-np.percentile(z,2))/2;yc[i]=(y.max()+y.min())/2
ok=~np.isnan(zc)
for arr in (zc,hz,yc):arr[~ok]=np.interp(bins[~ok],bins[ok],arr[ok])
def sample(img,x,y):
    x=np.clip(x,0,img.shape[1]-1.001);y=np.clip(y,0,img.shape[0]-1.001);x0=x.astype(int);y0=y.astype(int);fx=(x-x0)[:,None];fy=(y-y0)[:,None]
    return (img[y0,x0]*(1-fx)+img[y0,x0+1]*fx)*(1-fy)+(img[y0+1,x0]*(1-fx)+img[y0+1,x0+1]*fx)*fy
def proj(V,pts):
    s=sx(np.abs(pts[:,0]));ix=np.interp(s,[0,1,2,3,4],V['xs'])
    ix=np.where(s<0,V['xs'][0]+s*(V['xs'][1]-V['xs'][0]),ix)
    zn=(pts[:,2]-np.interp(s,bins,zc))/np.maximum(np.interp(s,bins,hz),1e-6)
    xi=np.clip(np.round(ix).astype(int),0,699);t=V['top'][xi];bo=V['bot'][xi]
    iy=(t+bo)/2-np.clip(zn,-1,1)*(bo-t)/2*0.9      # +z (frente, lado do polegar) = parte de cima da imagem
    return sample(SRC,ix,iy)
OUT=TEX.copy();WACC=np.zeros((TH,TW),np.float32);COL=np.zeros((TH,TW,3),np.float32)
tris=I.reshape(-1,3);tsel=sel[tris].sum(1)>=2
for a,bb,c in tris[tsel]:
    uv=UV[[a,bb,c]]*[TW,TH];xs,ys=uv[:,0],uv[:,1]
    x0,x1=max(0,int(xs.min())-1),min(TW-1,int(xs.max())+1);y0,y1=max(0,int(ys.min())-1),min(TH-1,int(ys.max())+1)
    gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
    den=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
    if abs(den)<1e-9:continue
    l0=((ys[1]-ys[2])*(gx-xs[2])+(xs[2]-xs[1])*(gy-ys[2]))/den;l1=((ys[2]-ys[0])*(gx-xs[2])+(xs[0]-xs[2])*(gy-ys[2]))/den
    e=-1.0/np.sqrt(abs(den)+1);mk=(l0>=e)&(l1>=e)&(1-l0-l1>=e)
    if not mk.any():continue
    L=np.stack([l0[mk],l1[mk],1-l0[mk]-l1[mk]],1);pts=L@P[[a,bb,c]];nr=L@N[[a,bb,c]];nr/=np.linalg.norm(nr,axis=1,keepdims=True)+1e-9
    s=sx(np.abs(pts[:,0]))
    ny=nr[:,1]-0.0  # dorso (mão virada para baixo na pose T): normal para cima
    wt=np.clip((ny+0.35)/0.7,0,1);wt=wt*wt*(3-2*wt)
    col=proj(VIEWS['dorso'],pts)*wt[:,None]+proj(VIEWS['palma'],pts)*(1-wt[:,None])
    fade=np.clip((s+0.15)/0.35,0,1)*np.clip((3.15-s)/0.35,0,1)   # entra depois do ombro e sai nos dedos
    yy,xx=np.nonzero(mk);yy+=y0;xx+=x0
    COL[yy,xx]=col;WACC[yy,xx]=np.maximum(WACC[yy,xx],fade)
# só onde a textura original é pele (não pinta braçadeiras nem ombreira)
mxT=TEX.max(2);mnT=TEX.min(2)
skin=np.clip((mxT-90)/40,0,1)*np.clip(((TEX[...,0]-TEX[...,2])-10)/20,0,1)
skin=np.asarray(Image.fromarray((skin*255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.5)),np.float32)/255
w=WACC*skin
# ajusta o tom da foto ao da pele do modelo (média e contraste por canal, nas áreas pintadas)
m=w>0.5
if m.any():
    mi=COL[m].mean(0);si=COL[m].std(0);mo=TEX[m].mean(0);so=TEX[m].std(0)
    k=np.clip(so/si,0.85,1.3);print('tom foto',mi.round(),'modelo',mo.round(),'k',k.round(2))
    COL=(COL-mi)*k+mo
OUT=TEX*(1-w[...,None])+COL*w[...,None]
Image.fromarray(OUT.clip(0,255).astype(np.uint8)).save('mix_tex_bracos.png')
print('texels pintados',int((w>0.5).sum()))
