"""Gera bracos_humanos.glb: recorta os braços de aethelred_bracos_humanos_leve.glb (o mesmo personagem gerado com
braços humanos, sem esqueleto) e os prende ao esqueleto de aethelred_rig_leve.glb.
1) alinha o modelo novo ao atual (escala, rotação e posição) pelo corpo inteiro (ICP);
2) acha os braços pela cor da pele e pela distância à cadeia ombro-cotovelo-punho-mão do esqueleto;
3) dá a cada vértice pesos nos ossos Shoulder/Arm/ForeArm/Hand (mistura nas juntas);
4) grava um GLB só com os braços e a textura, com as matrizes de ligação em extras.inverseBind.
Uso: python3 ferramentas/encaixar_bracos_humanos.py (precisa de numpy, scipy e Pillow)."""
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
import json,struct,numpy as np
def loadglb(F):
    d=open(F,'rb').read();l=struct.unpack('<I',d[12:16])[0];j=json.loads(d[20:20+l]);b0=20+l+8
    def acc(i):
        a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=b0+bv.get('byteOffset',0)+a.get('byteOffset',0)
        ct={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}[a['componentType']];n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        es=np.dtype(ct).itemsize*n;st=bv.get('byteStride',es)
        if st==es:x=np.frombuffer(d,dtype=ct,count=a['count']*n,offset=off).reshape(a['count'],n)
        else:
            raw=np.frombuffer(d,dtype=np.uint8,count=st*(a['count']-1)+es,offset=off)
            x=np.lib.stride_tricks.as_strided(raw,shape=(a['count'],es),strides=(st,1)).copy().view(ct).reshape(a['count'],n)
        return x
    return d,j,b0,acc
d2,j2,b02,acc2=loadglb(os.path.join(ROOT,'aethelred_bracos_humanos_leve.glb'))
p2=j2['meshes'][0]['primitives'][0]
P2=acc2(p2['attributes']['POSITION']).astype(np.float64);UV2=acc2(p2['attributes']['TEXCOORD_0']);N2=acc2(p2['attributes']['NORMAL']).astype(np.float64)
I2=acc2(p2['indices']).reshape(-1,3).astype(int)
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rr
rng=np.random.default_rng(0)
A=P[rng.choice(len(P),20000,replace=False)];B=P2[rng.choice(len(P2),20000,replace=False)]
tA=cKDTree(A)
def sim(X,Y):
    mx=X.mean(0);my=Y.mean(0);Xc=X-mx;Yc=Y-my;U,S,Vt=np.linalg.svd(Xc.T@Yc);D=np.eye(3);D[2,2]=np.sign(np.linalg.det(U@Vt))
    R=(U@D@Vt).T;s=(S*np.diag(D)).sum()/(Xc**2).sum();t=my-s*R@mx;return s,R,t
best=None
hA=P[:,1].max()-P[:,1].min();hB=P2[:,1].max()-P2[:,1].min()
for yaw in [0,180]:
  for tilt in [-30,-20,-10,0,10,20,30]:
    R0=Rr.from_euler('yx',[yaw,tilt],degrees=True).as_matrix();s=hA/hB
    Bc=B-B.mean(0);X=(s*(R0@Bc.T)).T+A.mean(0)
    s_,R_,t_=s,R0,A.mean(0)-s*R0@B.mean(0)
    for it in range(40):
        Xt=(s_*(R_@B.T)).T+t_;dd,ii=tA.query(Xt);keep=dd<np.percentile(dd,80)
        s_,R_,t_=sim(B[keep],A[ii[keep]])
    Xt=(s_*(R_@B.T)).T+t_;dd,_=tA.query(Xt);sc=np.median(dd)
    if best is None or sc<best[0]:best=(sc,s_,R_,t_,yaw,tilt)
sc,s_,R_,t_,yaw,tilt=best;print('best',yaw,tilt,sc,s_,Rr.from_matrix(R_).as_euler('yxz',degrees=True))
Q2=(s_*(R_@P2.T)).T+t_
Ral=R_

# cor da textura do modelo novo em cada vértice; pele: clara, avermelhada e pouco saturada
from PIL import Image;import io
img=j2['images'][0];bv=j2['bufferViews'][img['bufferView']];o=b02+bv.get('byteOffset',0)
tex=np.asarray(Image.open(io.BytesIO(d2[o:o+bv['byteLength']])).convert('RGB')).astype(np.float64)/255
H,Wd=tex.shape[:2];px=np.clip((UV2[:,0]%1)*Wd,0,Wd-1).astype(int);py=np.clip((UV2[:,1]%1)*H,0,H-1).astype(int)
C2=tex[py,px];r,g,b=C2.T;mx_=C2.max(1);mn_=C2.min(1);sat=(mx_-mn_)/np.maximum(mx_,1e-6)
skin=(r>.3)&(r>=g)&(g>=b*.95)&(sat>.12)&(sat<.6)&((r-g)<.22)&((g-b)>.02)
import scipy.sparse as sp
from scipy.spatial import cKDTree
nrm=np.linalg.norm
val2=C2.max(1)
Nw=(Ral@N2.T).T;Nw/=np.maximum(nrm(Nw,axis=1),1e-9)[:,None]
N=len(Q2)
A=sp.coo_matrix((np.ones(I2.size*2),(I2[:,[0,1,2,1,2,0]].ravel(),I2[:,[1,2,0,0,1,2]].ravel())),shape=(N,N)).tocsr()
info={}
sel_all=np.zeros(N,bool);Jn=np.zeros((N,4),int);Wn=np.zeros((N,4))
IBn=IB.copy()
def rot_between(a,b):
    a=a/nrm(a);b=b/nrm(b);v=np.cross(a,b);c=a@b
    if nrm(v)<1e-9:return np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);return np.eye(3)+vx+vx@vx*(1/(1+c))
for side,sg in [('Left',1),('Right',-1)]:
    S=BP[bi[side+'Arm']];Eo=BP[bi[side+'ForeArm']];Wo=BP[bi[side+'Hand']];Mo=BP[bi[side+'HandMiddle1']]
    L1=nrm(Eo-S);L2=nrm(Wo-Eo);L3=nrm(Mo-Wo)
    m=skin&(Q2[:,0]*sg>.12)&(Q2[:,1]<.45)&(Q2[:,1]>-.45)
    pts=Q2[m]
    # curva central: fatias ao longo da descida, centróide de cada fatia
    # curva central suave: x(y) e z(y) por polinômio de grau 3 sobre os pontos de pele
    yy=pts[:,1];cx=np.polyfit(yy,pts[:,0],3);cz=np.polyfit(yy,pts[:,2],3)
    ylo=np.percentile(yy,.5)
    ygrid=np.linspace(S[1]-.04,ylo,200);curve=np.stack([np.polyval(cx,ygrid),ygrid,np.polyval(cz,ygrid)],1)
    curve=np.vstack([S,curve])
    seg=np.linalg.norm(np.diff(curve,axis=0),axis=1);arc=np.concatenate([[0],np.cumsum(seg)])
    def at(s):
        k=np.searchsorted(arc,s)-1;k=np.clip(k,0,len(seg)-1);u=(s-arc[k])/seg[k];return curve[k]+(curve[k+1]-curve[k])*u
    En=at(L1)
    HANDLEN=.19
    Wn_=at(arc[-1]-HANDLEN)
    low=curve[-1];Hd=low-Wn_
    Mn=Wn_+Hd/nrm(Hd)*L3
    print(side,'S',S.round(3),'E',En.round(3),'(velho',Eo.round(3),') W',Wn_.round(3),'(velho',Wo.round(3),')',' arc',arc[-1].round(3),'L1+L2',(L1+L2).round(3))
    En,Wn_,Mn=Eo,Wo,Mo;Hd=Mo-Wo  # mesma pose do modelo atual: usa as juntas do esqueleto como estão
    info[side]=(S,En,Wn_,Mn)
    # seleção do braço: tudo perto da cadeia S-E-W-M (raio generoso) e além do plano do ombro
    chain=[S,En,Wn_,Mn+Hd/nrm(Hd)*.06]
    best=np.full(N,9.);T=np.zeros(N);segi=np.zeros(N,int)
    for k in range(3):
        a,b=chain[k],chain[k+1];ab=b-a;L=nrm(ab);u=ab/L;t=(Q2-a)@u;tc=np.clip(t,0,L);r=nrm(Q2-(a+np.outer(tc,u)),axis=1)
        if k==0:r=np.where(t<-.03,9,r)
        s_=r<best;best[s_]=r[s_];T[s_]=tc[s_]/L;segi[s_]=k
    candNV=(best<np.where(segi==2,.08,.085))&(Q2[:,0]*sg>.1)
    cand=candNV&(val2>.2)
    # cresce a partir da pele, só dentro do candidato
    reg=m&cand
    for _ in range(12):reg=(reg|((A@reg.astype(float))>0))&cand
    # fecha buracos (pele escura na textura): dilata e depois erode dentro da região candidata sem filtro de cor
    R1=reg.copy()
    for _ in range(3):R1=(R1|((A@R1.astype(float))>0))&candNV
    out=~R1
    for _ in range(3):out=out|((A@out.astype(float))>0)
    reg=reg|~out
    # perto do ombro só pele (não pegar alça/capa)
    sel_all|=reg
    # pesos: Arm / ForeArm / Hand com mistura nas juntas; topo do braço mistura com o Shoulder
    ids=[bi[side+n] for n in ['Shoulder','Arm','ForeArm','Hand']]
    v=np.nonzero(reg)[0]
    # parâmetro contínuo ao longo da cadeia em comprimento: 0..L1 braço, L1..L1+L2 antebraço, depois mão
    pos=np.where(segi[v]==0,T[v]*L1,np.where(segi[v]==1,L1+T[v]*L2,L1+L2+T[v]*(L3+.06)))
    bl=.035
    def ramp(x,c):return np.clip((x-(c-bl))/(2*bl),0,1)
    wE=ramp(pos,L1);wW=ramp(pos,L1+L2)
    wSh=np.clip(1-pos/.07,0,1)*.5
    w_arm=(1-wE)*(1-wSh);w_sh=(1-wE)*wSh;w_fa=wE*(1-wW);w_h=wW
    Jn[v]=np.array(ids)[None,:];Wn[v]=np.stack([w_sh,w_arm,w_fa,w_h],1)
    # matrizes de ligação novas: cada osso gira em torno da junta velha para a direção do segmento novo e vai para a junta nova
    for bn,(oj,oc),(nj,nc) in [('Arm',(S,Eo),(S,En)),('ForeArm',(Eo,Wo),(En,Wn_)),('Hand',(Wo,Mo),(Wn_,Mn))]:
        R3=rot_between(oc-oj,nc-nj);i=bi[side+bn]
        Mold=BW[i];Mnew=np.eye(4);Mnew[:3,:3]=R3@Mold[:3,:3];Mnew[:3,3]=nj
        IBn[i]=np.linalg.inv(Mnew)
    # dedos e polegar acompanham a mão (sem animação própria de dedos no braço novo)
    hi=bi[side+'Hand']
    for f in ['Thumb','Index','Middle','Ring','Pinky']:
        for k in (1,2,3):IBn[bi[f'{side}Hand{f}{k}']]=IBn[hi]
Wn/=np.maximum(Wn.sum(1),1e-9)[:,None]
tri=I2[sel_all[I2].all(1)]
used=np.unique(tri);remap=-np.ones(N,int);remap[used]=np.arange(len(used))
print('verts braço novo',len(used),'tris',len(tri))
z=dict(P=Q2[used],N=Nw[used],UV=UV2[used],J=Jn[used],W=Wn[used],I=remap[tri],IB=IBn)
img=j2['images'][0];bv=j2['bufferViews'][img['bufferView']];jpg=d2[b02+bv.get('byteOffset',0):b02+bv.get('byteOffset',0)+bv['byteLength']]
P=z['P'].astype(np.float32);Nn=z['N'].astype(np.float32);UV=z['UV'].astype(np.float32);Jt=z['J'].astype(np.uint16);Wt=z['W'].astype(np.float32);I=z['I'].astype(np.uint32).ravel()
blobs=[];views=[];accs=[]
def add(arr,typ,ct,target=None,minmax=False):
    off=sum(len(b) for b in blobs);b=arr.tobytes();pad=(4-len(b)%4)%4;blobs.append(b+b'\0'*pad)
    v={'buffer':0,'byteOffset':off,'byteLength':len(b)}
    if target:v['target']=target
    views.append(v);a={'bufferView':len(views)-1,'componentType':ct,'count':len(arr),'type':typ}
    if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
    accs.append(a);return len(accs)-1
iP=add(P,'VEC3',5126,34962,True);iN=add(Nn,'VEC3',5126,34962);iU=add(UV,'VEC2',5126,34962);iJ=add(Jt,'VEC4',5123,34962);iW=add(Wt,'VEC4',5126,34962);iI=add(I,'SCALAR',5125,34963)
off=sum(len(b) for b in blobs);pad=(4-len(jpg)%4)%4;blobs.append(jpg+b'\0'*pad);views.append({'buffer':0,'byteOffset':off,'byteLength':len(jpg)})
binb=b''.join(blobs)
gl={'asset':{'version':'2.0','generator':'aethelred bracos humanos'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'bracos_humanos','mesh':0}],
 'meshes':[{'name':'bracos_humanos','primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU,'JOINTS_0':iJ,'WEIGHTS_0':iW},'indices':iI,'material':0}],
   'extras':{'inverseBind':[float(x) for x in z['IB'].transpose(0,2,1).ravel()]}}],
 'materials':[{'name':'pele','doubleSided':True,'pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':0,'roughnessFactor':.8}}],
 'textures':[{'source':0}],'images':[{'bufferView':len(views)-1,'mimeType':'image/jpeg'}],
 'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(binb)}]}
js=json.dumps(gl,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb
F=os.path.join(ROOT,'bracos_humanos.glb')
open(F,'wb').write(out);print(F,len(out))
