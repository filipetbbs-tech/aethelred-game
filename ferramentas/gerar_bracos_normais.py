"""Gera bracos_normais.bin a partir de aethelred_rig_leve.glb: a forma normal (humana) dos braços.
Recolhe os espetos de osso do braço/antebraço/mão para a superfície da pele, corta as garras além da
última falange e transforma os dedos em tubos arredondados; as partes de osso ganham uma cor que as tinge
para o tom da pele. Uso: python3 ferramentas/gerar_bracos_normais.py (precisa de numpy, scipy e Pillow)."""
import json,struct,numpy as np
import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F=os.path.join(ROOT,'aethelred_rig_leve.glb')
d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b0=20+l+8
def acc(i):
    a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=b0+bv.get('byteOffset',0)+a.get('byteOffset',0)
    ct={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    es=np.dtype(ct).itemsize*n;st=bv.get('byteStride',es)
    if st==es:x=np.frombuffer(d,dtype=ct,count=a['count']*n,offset=off).reshape(a['count'],n)
    else:
        raw=np.frombuffer(d,dtype=np.uint8,count=st*(a['count']-1)+es,offset=off)
        x=np.lib.stride_tricks.as_strided(raw,shape=(a['count'],es),strides=(st,1)).copy().view(ct).reshape(a['count'],n)
    if a.get('normalized'):x=x.astype(np.float32)/np.iinfo(ct).max
    return x
p=j['meshes'][0]['primitives'][0]
P=acc(p['attributes']['POSITION']).astype(np.float64)
UV=acc(p['attributes']['TEXCOORD_0'])
J=acc(p['attributes']['JOINTS_0']).astype(int);W=acc(p['attributes']['WEIGHTS_0']).astype(np.float64)
I=acc(p['indices']).reshape(-1,3).astype(int) if 'indices' in p else None
sk=j['skins'][0];names=[j['nodes'][i]['name'].replace('mixamorig','') for i in sk['joints']]
IB=acc(sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1).astype(np.float64)
BW=np.linalg.inv(IB);BP=BW[:,:3,3]
bi={n:i for i,n in enumerate(names)}
dom=J[np.arange(len(J)),W.argmax(1)]

# cor da textura em cada vértice (para achar o que é osso: claro e pouco saturado; pele: avermelhada)
from PIL import Image
import io
img=j['images'][0];bv=j['bufferViews'][img['bufferView']];o=b0+bv.get('byteOffset',0)
tex=np.asarray(Image.open(io.BytesIO(d[o:o+bv['byteLength']])).convert('RGB')).astype(np.float64)/255
H,Wd=tex.shape[:2]
px=np.clip((UV[:,0]%1)*Wd,0,Wd-1).astype(int);py=np.clip((UV[:,1]%1)*H,0,H-1).astype(int)
Ccol=tex[py,px]
import colorsys,scipy.sparse as sp
C=Ccol;N=len(P)
NR=acc(p['attributes']['NORMAL']).astype(np.float64)
mx=C.max(1);mn=C.min(1);sat=np.where(mx>0,(mx-mn)/np.maximum(mx,1e-6),0);val=mx
bone=np.clip((.33-sat)/.13,0,1)*np.clip((val-.18)/.12,0,1)
# suaviza pela malha
A=sp.coo_matrix((np.ones(I.size*2),(np.repeat(I,3,axis=0).ravel() if False else I[:,[0,1,2,1,2,0]].ravel(),I[:,[1,2,0,0,1,2]].ravel())),shape=(N,N)).tocsr()
deg=np.asarray(A.sum(1)).ravel()+1
for _ in range(2):bone=(A@bone+bone)/deg
FING=['Thumb','Index','Middle','Ring','Pinky']
def wsum(side,keys):
    ids=[bi[side+k] for k in keys if side+k in bi];w=np.zeros(N)
    for i in ids:w+=(W*(J==i)).sum(1)
    return w
Q=P.copy();tint=np.ones((N,3))
skinCol=None
nrm=np.linalg.norm
for side in ['Left','Right']:
    chain=['Arm','ForeArm','Hand']
    wArm=wsum(side,['Arm','ForeArm','Hand']);wF=wsum(side,[f'Hand{f}{k}' for f in FING for k in (1,2,3)])
    wAll=wArm+wF
    fa=wsum(side,['ForeArm'])>.5
    sk=np.median(C[fa&(bone<.2)],0)
    if skinCol is None:skinCol=sk
    # --- espetos: polilinha Arm -> ForeArm -> Hand -> Middle1 ---
    pts=[BP[bi[side+n]] for n in ['Arm','ForeArm','Hand','HandMiddle1']]
    m=(wArm>.35)
    v=P[m];best=np.full(len(v),1e9);T=np.zeros(len(v));PR=np.zeros_like(v);U=np.zeros_like(v)
    acc_len=0;lens=[]
    for k in range(3):
        a,b=pts[k],pts[k+1];ab=b-a;L=nrm(ab);u=ab/L;t=np.clip((v-a)@u,0,L);pr=a+np.outer(t,u);r=nrm(v-pr,axis=1)
        sel=r<best;best[sel]=r[sel];T[sel]=acc_len+t[sel];PR[sel]=pr[sel];U[sel]=u
        acc_len+=L
    r=best;dirv=(v-PR)/np.maximum(r,1e-9)[:,None]
    # setor angular em torno do eixo, com referência fixa (para frente do corpo, z)
    ref=np.array([0,0,1.]);e1=ref-np.outer(U@ref,1)*U;e1/=nrm(e1,axis=1)[:,None];e2=np.cross(U,e1)
    ang=np.arctan2((dirv*e2).sum(1),(dirv*e1).sum(1))
    tb=(T/acc_len*24).astype(int);ab_=((ang+np.pi)/(2*np.pi)*8).astype(int)%8
    R=np.zeros(len(v));bm=bone[m]
    for key in set(zip(tb,ab_)):
        s=(tb==key[0])&(ab_==key[1])
        ok=s&(bm<.35)
        if ok.sum()<4:
            ok=(np.abs(tb-key[0])<=1)&(bm<.35)
        R[s]=np.percentile(r[ok],68) if ok.sum()>=3 else np.percentile(r[s],40)
    lim=R*1.02
    rn=np.where(r>lim,lim+(r-lim)*.03,r)
    new=PR+dirv*rn[:,None]
    ww=np.clip((wArm[m]-.35)/.35,0,1)[:,None]*np.clip(.6+bm*1.5,0,1)[:,None]
    Q[m]=Q[m]*(1-ww)+new*ww
    # --- dedos: corta a garra além da última falange, engrossa e arredonda a ponta ---
    for f in FING:
        f1,f2,f3=[BP[bi[f'{side}Hand{f}{k}']] for k in (1,2,3)]
        ids=[f'Hand{f}{k}' for k in (1,2,3)];w=wsum(side,ids);m=w>.2
        u=f3-f2;Ls=nrm(u);u/=Ls
        v=Q[m];
        # eixo do dedo como polilinha f1-f2-f3-ponta
        s=(v-f3)@u
        tipL=Ls*.55
        sn=np.where(s>tipL*.5,tipL*.5+(np.minimum(s,Ls*3)-tipL*.5)*.12,s)
        proj=f3+np.outer(s,u);rv=v-proj;r=nrm(rv,axis=1)
        # antes de f3 o eixo é o segmento correspondente
        for a,b in [(f1,f2),(f2,f3)]:
            ab=b-a;L=nrm(ab);uu=ab/L;t=(v-a)@uu;sel=(t>=0)&(t<=L)&(s<0)
            pr=a+np.outer(t,uu);rr=nrm(v-pr,axis=1);rv[sel]=(v-pr)[sel];r[sel]=rr[sel];proj[sel]=pr[sel]
        rmin=.0105
        tip=np.clip(sn/(tipL*.62),0,1)
        rt=np.maximum(r,rmin)*np.sqrt(np.maximum(1-tip**2,.08))
        rt=np.minimum(rt,.016)
        base=np.where((s>0)[:,None],f3+np.outer(sn,u),proj)
        dr=rv/np.maximum(r,1e-9)[:,None]
        ax=np.where((s>0)[:,None],u[None,:],np.zeros(3)[None,:])
        nn=NR[m]-((NR[m]*u).sum(1))[:,None]*u;nn/=np.maximum(nrm(nn,axis=1),1e-9)[:,None]
        # dedo vira um tubo: a parte radial vai até o raio do perfil e a normal da face empurra para fora,
        # transformando a lâmina achatada da garra num dedo arredondado
        rr=np.minimum(r,rt*.75)
        new=base+dr*rr[:,None]+nn*(rt*.8)[:,None]
        ww=np.clip((w[m]-.2)/.3,0,1)[:,None]
        Q[m]=Q[m]*(1-ww)+new*ww
    # --- cor: partes de osso (e todos os dedos) tingidas para o tom da pele ---
    amt=np.clip(np.maximum(bone*1.6,np.clip(wF,0,1)*.9),0,1)*np.clip(wAll*1.5,0,1)
    t_=np.clip(sk[None,:]/np.maximum(C,1e-3),0,1.0)
    tint=np.where((wAll>.05)[:,None],tint*(1-amt[:,None])+t_*amt[:,None],tint)
D=nrm(Q-P,axis=1);ch=(D>1e-4)|(np.abs(tint-1).max(1)>.01)
idx=np.nonzero(ch)[0].astype(np.uint32)
print('vertices',len(idx),'moved',(D>1e-3).sum(),'maxD',D.max().round(3),'skin',skinCol.round(2))
delta=(Q-P)[idx].astype(np.float32);tc=np.clip(tint[idx]*255,0,255).astype(np.uint8)
hdr=np.array([len(idx)],np.uint32)
open(os.path.join(ROOT,'bracos_normais.bin'),'wb').write(hdr.tobytes()+idx.tobytes()+delta.tobytes()+tc.tobytes())
print('bytes',4+len(idx)*19)
