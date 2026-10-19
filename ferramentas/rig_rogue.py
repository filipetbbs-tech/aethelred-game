"""Gera aethelred_humano_rig.glb a partir de rogue_warrior_leve.glb (modelo humano sem esqueleto, com texturas PBR:
cor, normal e metal/rugosidade), preso ao esqueleto de aethelred_rig_leve.glb.
1) alinha o modelo ao atual pelo corpo (ICP, sem os braços);
2) acha a cadeia de cada braço pela cor da pele (ombro-cotovelo-punho-mão) e gira os ossos do braço para essa pose
   (matrizes de ligação próprias, extras.inverseBind);
3) braços: pesos pela posição ao longo da cadeia (mão inteira presa ao osso da mão);
   capa (vermelho saturado): pesos só no tronco (Hips/Spine/Spine1/Spine2 pela altura), nunca nas pernas;
   resto do corpo: pesos dos pontos mais próximos do modelo atual (sem braços e sem capa dele).
Uso: python3 ferramentas/rig_rogue.py (precisa de numpy, scipy e Pillow)."""
import os,sys,json,struct,io,base64,numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rr
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nrm=np.linalg.norm
def loadglb(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b0=20+l+8
    def acc(i):
        a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=b0+bv.get('byteOffset',0)+a.get('byteOffset',0)
        ct={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32,5122:np.int16,5120:np.int8}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        es=np.dtype(ct).itemsize*n;st=bv.get('byteStride',es)
        if st==es:x=np.frombuffer(d,dtype=ct,count=a['count']*n,offset=off).reshape(a['count'],n)
        else:
            raw=np.frombuffer(d,dtype=np.uint8,count=st*(a['count']-1)+es,offset=off)
            x=np.lib.stride_tricks.as_strided(raw,shape=(a['count'],es),strides=(st,1)).copy().view(ct).reshape(a['count'],n)
        if a.get('normalized'):x=x.astype(np.float32)/np.iinfo(ct).max
        return x
    return d,j,b0,acc
def imgbytes(d,j,b0,k):
    img=j['images'][k];bv=j['bufferViews'][img['bufferView']];o=b0+bv.get('byteOffset',0);return d[o:o+bv['byteLength']],img.get('mimeType','image/jpeg')
def sample(tex,uv):
    H,Wd=tex.shape[:2];return tex[np.clip((uv[:,1]%1)*H,0,H-1).astype(int),np.clip((uv[:,0]%1)*Wd,0,Wd-1).astype(int)]
# --- modelo atual (rigado) ---
d,j,b0,acc=loadglb(os.path.join(ROOT,'aethelred_rig_leve.glb'));p=j['meshes'][0]['primitives'][0]
P=acc(p['attributes']['POSITION']).astype(np.float64);J=acc(p['attributes']['JOINTS_0']).astype(int);W=acc(p['attributes']['WEIGHTS_0']).astype(np.float64)
UVo=acc(p['attributes']['TEXCOORD_0'])
sk=j['skins'][0];names=[j['nodes'][i]['name'].replace('mixamorig','') for i in sk['joints']];bi={n:i for i,n in enumerate(names)}
IB=acc(sk['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1).astype(np.float64);BW=np.linalg.inv(IB);BP=BW[:,:3,3]
Co=sample(np.asarray(Image.open(io.BytesIO(imgbytes(d,j,b0,0)[0])).convert('RGB')).astype(np.float64)/255,UVo)
def ws(keys):
    w=np.zeros(len(P))
    for k in keys:w+=(W*(J==bi[k])).sum(1)
    return w
armBones=[n for n in names if any(s in n for s in ['Arm','Hand']) and 'Shoulder' not in n]
armw=ws(armBones)
# --- modelo novo ---
dR,jR,bR,accR=loadglb(os.path.join(ROOT,'rogue_warrior_leve.glb'));pR=jR['meshes'][0]['primitives'][0]
PR=accR(pR['attributes']['POSITION']).astype(np.float64);NR=accR(pR['attributes']['NORMAL']).astype(np.float64)
UVR=accR(pR['attributes']['TEXCOORD_0']).astype(np.float32);IR=accR(pR['indices']).reshape(-1,3).astype(int)
matR=jR['materials'][0];texIdx=lambda t:jR['textures'][t['index']]['source']
CR=sample(np.asarray(Image.open(io.BytesIO(imgbytes(dR,jR,bR,texIdx(matR['pbrMetallicRoughness']['baseColorTexture']))[0])).convert('RGB')).astype(np.float64)/255,UVR)
# --- 1) alinhamento ---
rng=np.random.default_rng(0)
A=P[armw<.2];A=A[rng.choice(len(A),20000,replace=False)];tA=cKDTree(A);B=PR[rng.choice(len(PR),20000,replace=False)]
def sim(X,Y):
    mx=X.mean(0);my=Y.mean(0);Xc=X-mx;Yc=Y-my;U,S,Vt=np.linalg.svd(Xc.T@Yc);D=np.eye(3);D[2,2]=np.sign(np.linalg.det(U@Vt))
    R=(U@D@Vt).T;s=(S*np.diag(D)).sum()/(Xc**2).sum();t=my-s*R@mx;return s,R,t
hA=P[:,1].max()-P[:,1].min();hB=PR[:,1].max()-PR[:,1].min();best=None
for tilt in [-30,-22,-10]:
    R0=Rr.from_euler('x',tilt,degrees=True).as_matrix();s_=hA/hB;R_=R0;t_=A.mean(0)-s_*R0@B.mean(0)
    for it in range(50):
        Xt=(s_*(R_@B.T)).T+t_;dd,ii=tA.query(Xt);keep=dd<np.percentile(dd,75);s_,R_,t_=sim(B[keep],A[ii[keep]])
    dd,_=tA.query((s_*(R_@B.T)).T+t_);sc=np.median(dd)
    if best is None or sc<best[0]:best=(sc,s_,R_,t_)
sc,s_,R_,t_=best;print('alinhamento',round(sc,4),round(s_,3))
Q=(s_*(R_@PR.T)).T+t_;NQ=(R_@NR.T).T;NQ/=np.maximum(nrm(NQ,axis=1),1e-9)[:,None]
# --- cores: pele e capa ---
r,g,b=CR.T;mx_=CR.max(1);mn_=CR.min(1);sat=(mx_-mn_)/np.maximum(mx_,1e-6)
skin=(r>.4)&(r>=g)&(g>=b*.95)&(sat>.12)&(sat<.55)&((r-g)<.25)
def adj(X,T):
    q=np.round(X/1e-5).astype(np.int64);_,inv=np.unique(q,axis=0,return_inverse=True);inv=inv.ravel();Tw=inv[T];n=inv.max()+1
    return sp.coo_matrix((np.ones(Tw.size*2),(Tw[:,[0,1,2,1,2,0]].ravel(),Tw[:,[1,2,0,0,1,2]].ravel())),shape=(n,n)).tocsr(),inv,n
Aq,inv,nq=adj(Q,IR);dg=np.asarray(Aq.sum(1)).ravel()+1
redsc=np.clip((r/np.maximum(g,1e-3)-1.25)/.5,0,1)*np.clip((sat-.3)/.2,0,1)
sc_=np.zeros(nq);cn=np.zeros(nq);np.add.at(sc_,inv,redsc);np.add.at(cn,inv,1);sc_/=cn
for _ in range(15):sc_=(Aq@sc_+sc_)/dg
# capa = tecido fino (a face de trás está a menos de 2,5 cm, procurando vizinhos atrás de cada ponto, na direção
# oposta à normal) E vermelho; a armadura de escamas também é avermelhada mas tem volume por dentro
import trimesh
vn=trimesh.Trimesh(Q,IR,process=False).vertex_normals;tQ=cKDTree(Q);thick=np.full(len(Q),9.)
for s0 in range(0,len(Q),4000):
    for k_,lst in enumerate(tQ.query_ball_point(Q[s0:s0+4000],.05)):
        i=s0+k_
        if len(lst)<2:continue
        dv=Q[lst]-Q[i];al=-(dv@vn[i]);lat=np.sqrt(np.maximum((dv*dv).sum(1)-al**2,0))
        ok=(al>.002)&(lat<.008+al*.15)&((vn[lst]@vn[i])<-.2)
        if ok.any():thick[i]=al[ok].min()
s0_=(thick<.025).astype(float)*np.clip(redsc*1.5,0,1)
sc_=np.zeros(nq);cn=np.zeros(nq);np.add.at(sc_,inv,s0_);np.add.at(cn,inv,1);sc_/=cn
for _ in range(20):sc_=(Aq@sc_+sc_)/dg
mm=sc_>.2
for _ in range(3):mm=mm|((Aq@mm.astype(float))>0)
oo=~mm
for _ in range(3):oo=oo|((Aq@oo.astype(float))>0)
cape=(~oo)[inv]
# --- 2) cadeias dos braços e matrizes de ligação ---
def rot_between(a,b_):
    a=a/nrm(a);b_=b_/nrm(b_);v=np.cross(a,b_);c=a@b_
    if nrm(v)<1e-9:return np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);return np.eye(3)+vx+vx@vx*(1/(1+c))
def at(curve,s):
    seg=nrm(np.diff(curve,axis=0),axis=1);arc=np.concatenate([[0],np.cumsum(seg)])
    k=np.clip(np.searchsorted(arc,s)-1,0,len(seg)-1);u=(s-arc[k])/seg[k];return curve[k]+(curve[k+1]-curve[k])*u,arc[-1]
IBn=IB.copy();N=len(Q);NB=len(names)
accW=np.zeros((N,NB));armSel=np.zeros(N,bool)
for side,sg in [('Left',1),('Right',-1)]:
    S=BP[bi[side+'Arm']];Eo=BP[bi[side+'ForeArm']];Wo=BP[bi[side+'Hand']];Mo=BP[bi[side+'HandMiddle1']]
    L1=nrm(Eo-S);L2=nrm(Wo-Eo);L3=nrm(Mo-Wo)
    m=skin&~cape&(Q[:,0]*sg>.12)&(Q[:,1]<S[1]+.02)&(Q[:,1]>-.45);pts=Q[m];yy=pts[:,1]
    cx=np.polyfit(yy,pts[:,0],3);cz=np.polyfit(yy,pts[:,2],3)
    yg=np.linspace(min(S[1]-.03,np.percentile(yy,99.5)),np.percentile(yy,.5),200)
    curve=np.vstack([S,np.stack([np.polyval(cx,yg),yg,np.polyval(cz,yg)],1)]);_,total=at(curve,0)
    k=total/(L1+L2+.21);hand=.19*k;armlen=total-hand
    E,_=at(curve,armlen*L1/(L1+L2));Wn,_=at(curve,armlen);tip=curve[-1];Hd=(tip-Wn)/nrm(tip-Wn);Mn=Wn+Hd*L3*k
    print(side,'cotovelo',E.round(3),'punho',Wn.round(3))
    for bn,(oj,oc),(nj,nc) in [('Arm',(S,Eo),(S,E)),('ForeArm',(Eo,Wo),(E,Wn)),('Hand',(Wo,Mo),(Wn,Mn))]:
        R3=rot_between(oc-oj,nc-nj);i=bi[side+bn];Mnew=np.eye(4);Mnew[:3,:3]=R3@BW[i][:3,:3];Mnew[:3,3]=nj;IBn[i]=np.linalg.inv(Mnew)
    for f in ['Thumb','Index','Middle','Ring','Pinky']:
        for kk in (1,2,3):IBn[bi[f'{side}Hand{f}{kk}']]=IBn[bi[side+'Hand']]
    # seleção do braço: perto da cadeia nova, além do ombro, sem capa; bola na mão para o polegar
    chain=[S,E,Wn,tip+Hd*.03];best_=np.full(N,9.);pos=np.zeros(N);base=0
    for kq in range(3):
        a_,b_=chain[kq],chain[kq+1];L_=nrm(b_-a_);u_=(b_-a_)/L_;t=(Q-a_)@u_;tc=np.clip(t,0,L_);rr=nrm(Q-(a_+np.outer(tc,u_)),axis=1)
        if kq==0:rr=np.where(t<-.02,9,rr)
        s_m=rr<best_;best_[s_m]=rr[s_m];pos[s_m]=base+tc[s_m];base+=L_
    Hc=Wn+Hd*hand*.5;ball=nrm(Q-Hc,axis=1)<hand*.65
    # braço: perto da cadeia (raio de braço musculoso) e mais perto dela do que do eixo do tronco
    hipP=BP[bi['Hips']];nkP=BP[bi['Neck']];ua=(nkP-hipP)/nrm(nkP-hipP);dq=Q-hipP
    dSpine=nrm(dq-np.outer(dq@ua,ua),axis=1)
    sel=(((best_<.11)&(best_<dSpine*.75))|ball)&(Q[:,0]*sg>.08)&~cape
    Larm=nrm(E-S);Lfa=nrm(Wn-E);bl=.035
    wE=np.clip((pos-(Larm-bl))/(2*bl),0,1);wW=np.clip((pos-(Larm+Lfa-bl))/(2*bl),0,1);wSh=np.clip(1-pos/.07,0,1)*.5
    v=np.nonzero(sel)[0]
    accW[v]=0
    accW[v,bi[side+'Shoulder']]=((1-wE)*wSh)[v];accW[v,bi[side+'Arm']]=((1-wE)*(1-wSh))[v];accW[v,bi[side+'ForeArm']]=(wE*(1-wW))[v];accW[v,bi[side+'Hand']]=wW[v]
    armSel|=sel
# --- 3) capa: só tronco, pela altura ---
hip=BP[bi['Hips']];sp2=BP[bi['Spine2']];up=(sp2-hip)/nrm(sp2-hip)
hh=(Q-hip)@up/nrm(sp2-hip)
capeOnly=cape&~armSel
v=np.nonzero(capeOnly)[0];hv=hh[v]
for name_,lo,hi_ in [('Hips',-9,.3),('Spine',.3,.6),('Spine1',.6,.85),('Spine2',.85,9)]:
    accW[v,bi[name_]]=((hv>=lo)&(hv<hi_)).astype(float)
# abaixo do quadril a capa acompanha um pouco a coxa do mesmo lado (mais na aba da frente, que fica entre as pernas),
# para as pernas não a atravessarem sem esticá-la como antes
fw=np.cross(BP[bi['LeftArm']]-BP[bi['RightArm']],up);fw/=nrm(fw)
if (BP[bi['LeftToeBase']]-BP[bi['LeftFoot']])@fw<0:fw=-fw
lowv=hv<.3
for side,sg in [('Left',1),('Right',-1)]:
    sl=lowv&(Q[v,0]*sg>0)
    kq=np.where(((Q[v]-hip)@fw)>0,.3,.1)*np.clip((.3-hv)/.4,0,1)
    accW[v[sl],bi['Hips']]*=(1-kq[sl]);accW[v[sl],bi[side+'UpLeg']]+=kq[sl]
# suaviza a transição entre os ossos do tronco na capa
for _ in range(3):
    tmp=np.zeros((nq,NB));cnt=np.zeros(nq);np.add.at(tmp,inv[v],accW[v]);np.add.at(cnt,inv[v],1)
    tmp2=(Aq@tmp+tmp);c2=(Aq@cnt+cnt);tmp2/=np.maximum(c2,1e-9)[:,None];accW[v]=tmp2[inv[v]]
# --- resto do corpo: pesos dos vizinhos no modelo atual (sem braços e sem capa dele) ---
redO=(Co[:,0]>Co[:,1]*1.6)&(Co[:,0]>Co[:,2]*1.4)
src=(armw<.3)&~redO;srcI=np.nonzero(src)[0];tS=cKDTree(P[srcI])
rest=np.nonzero(~armSel&~capeOnly)[0]
dd,ii=tS.query(Q[rest],k=6);wk=1/np.maximum(dd,1e-4)**2;wk/=wk.sum(1,keepdims=True)
for kk in range(6):
    vi=srcI[ii[:,kk]]
    for c in range(4):np.add.at(accW,(rest,J[vi,c]),W[vi,c]*wk[:,kk])
top=np.argsort(-accW,axis=1)[:,:4];tw=np.take_along_axis(accW,top,1);tw/=np.maximum(tw.sum(1,keepdims=True),1e-9)
JF=top.astype(np.uint8);WF=np.clip(np.round(tw*255),0,255).astype(np.uint8);WF[:,0]+=(255-WF.astype(int).sum(1)).astype(np.uint8)
print('vertices',N,'braços',armSel.sum(),'capa',capeOnly.sum())
# --- grava ---
blobs=[];views=[];accs=[]
def add(arr,typ,ct,target=None,minmax=False,normalized=False):
    off=sum(len(x) for x in blobs);bb=arr.tobytes();blobs.append(bb+b'\0'*((4-len(bb)%4)%4))
    vv={'buffer':0,'byteOffset':off,'byteLength':len(bb)}
    if target:vv['target']=target
    views.append(vv);a={'bufferView':len(views)-1,'componentType':ct,'count':len(arr),'type':typ}
    if normalized:a['normalized']=True
    if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
    accs.append(a);return len(accs)-1
iP=add(Q.astype(np.float32),'VEC3',5126,34962,True);iN=add(NQ.astype(np.float32),'VEC3',5126,34962);iU=add(UVR,'VEC2',5126,34962)
iJ=add(JF,'VEC4',5121,34962);iW=add(WF,'VEC4',5121,34962,normalized=True);iI=add(IR.astype(np.uint32).ravel(),'SCALAR',5125,34963)
images=[]
for k in range(len(jR['images'])):
    bb,mt=imgbytes(dR,jR,bR,k);off=sum(len(x) for x in blobs);blobs.append(bb+b'\0'*((4-len(bb)%4)%4));views.append({'buffer':0,'byteOffset':off,'byteLength':len(bb)})
    images.append({'bufferView':len(views)-1,'mimeType':mt})
binb=b''.join(blobs)
mat=json.loads(json.dumps(matR));mat['name']='humano'
gl={'asset':{'version':'2.0','generator':'aethelred rig rogue'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'humano','mesh':0}],
 'meshes':[{'name':'humano','primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU,'JOINTS_0':iJ,'WEIGHTS_0':iW},'indices':iI,'material':0}],
   'extras':{'inverseBind':[float(x) for x in IBn.transpose(0,2,1).ravel()]}}],
 'materials':[mat],'textures':jR['textures'],'images':images,'samplers':jR.get('samplers',[{}]) if jR.get('samplers') else [],
 'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(binb)}]}
if not gl['samplers']:del gl['samplers']
js=json.dumps(gl,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb
F=os.path.join(ROOT,'aethelred_humano_rig.glb');open(F,'wb').write(out);print(F,len(out))
