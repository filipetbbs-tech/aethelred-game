"""Gera aethelred_corpo_rig.glb: o corpo modelos/rogue_warrior_t_pose_leve.glb (T-pose, sem esqueleto, texturas PBR)
preso ao esqueleto de aethelred_rig_leve.glb (só o esqueleto e as animações dele são usados; a malha antiga é descartada).
1) acha as juntas no corpo novo: proporções do esqueleto antigo escaladas pela altura, refinadas pela geometria
   (centro do tronco e das pernas por fatias, linha do braço em T, punho pela munhequeira, 4 dedos agrupados
   lado a lado e o polegar à frente);
2) cada osso é girado da pose antiga para a nova (mesma torção) e posto na junta nova: matrizes de ligação
   próprias (extras.inverseBind), tudo convertido para o espaço de ligação do esqueleto antigo;
3) pesos pela distância de cada ponto aos segmentos dos ossos (1/d^4 entre os mais próximos), suavizados pela
   superfície (menos nas mãos, para os dedos não se misturarem).
Uso: python3 ferramentas/rig_tpose.py"""
import os,sys,json,struct,io,numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from PIL import Image
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nrm=np.linalg.norm
exec("ROOT='%s'\n"%ROOT+open(os.path.join(ROOT,'ferramentas','esqueleto.py')).read().split("nrm=np.linalg.norm",1)[1])
def load_static(F):
    d,j,b0,acc=loadglb(F);p=j['meshes'][0]['primitives'][0]
    return dict(d=d,j=j,b0=b0,P=acc(p['attributes']['POSITION']).astype(np.float64),N=acc(p['attributes']['NORMAL']).astype(np.float64),
        UV=acc(p['attributes']['TEXCOORD_0']).astype(np.float32),I=acc(p['indices']).reshape(-1,3).astype(int),mat=j['materials'][0])
def rot_between(a,b_):
    a=a/nrm(a);b_=b_/nrm(b_);v=np.cross(a,b_);c=a@b_
    if nrm(v)<1e-9:return np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]);return np.eye(3)+vx+vx@vx*(1/(1+c))
# quadro "de pé" do esqueleto antigo
mU=(BP[bi['LeftUpLeg']]+BP[bi['RightUpLeg']])/2;mF=(BP[bi['LeftFoot']]+BP[bi['RightFoot']])/2
upO=(mU-mF)/nrm(mU-mF);rtO=BP[bi['LeftUpLeg']]-BP[bi['RightUpLeg']];rtO-=(rtO@upO)*upO;rtO/=nrm(rtO);fwO=np.cross(rtO,upO)
FO=np.stack([rtO,upO,fwO],1)
J0=(BP-mF)@FO;P0=(P-mF)@FO;g0=P0[:,1].min();top0=P0[:,1].max()
PARENT={};CHILD={}
for i,nidx in enumerate(sk['joints']):
    for c in j['nodes'][nidx].get('children',[]):
        if c in sk['joints']:PARENT[names[sk['joints'].index(c)]]=names[i]
for c,p_ in PARENT.items():CHILD.setdefault(p_,[]).append(c)
MAINCHILD={'Hips':'Spine','Spine2':'Neck','LeftHand':'LeftHandMiddle1','RightHand':'RightHandMiddle1'}
FING=['Thumb','Index','Middle','Ring','Pinky']
def fit_joints(B,C):
    """juntas do esqueleto no corpo em T (coordenadas do corpo: y para cima, frente +z, esquerda +x, chão em 0)"""
    PB=B['P'];k=(PB[:,1].max()-PB[:,1].min())/(top0-g0)
    NJ={n:(J0[bi[n]]-np.array([0,g0,0]))*k for n in names}
    def slicec(axis,val,w,mask):
        m=mask&(np.abs(PB[:,axis]-val)<w);return PB[m].mean(0) if m.sum()>10 else None
    for n in ['Hips','Spine','Spine1','Spine2','Neck','Head']:
        y=NJ[n][1];c=slicec(1,y,.02,np.abs(PB[:,0])<.25)
        if c is not None:NJ[n]=np.array([0,y,c[2]])
    r,g,b=C.T;mx=C.max(1);mn=C.min(1);sat=(mx-mn)/np.maximum(mx,1e-6);skin=(r>.45)&(r>=g)&(g>=b*.95)&(sat>.1)&(sat<.5)
    for side,sg in [('Left',1),('Right',-1)]:
        for n in ['UpLeg','Leg','Foot']:
            y=NJ[side+n][1];c=slicec(1,y,.015,(PB[:,0]*sg>0.02)&(np.abs(PB[:,0])<.35))
            if c is not None:NJ[side+n]=np.array([c[0],y,c[2]])
        ft=NJ[side+'Foot'];m=(PB[:,1]<.1)&(np.abs(PB[:,0]-ft[0])<.08);zt=np.percentile(PB[m][:,2],92)
        NJ[side+'ToeBase']=np.array([ft[0],.04,ft[2]+(zt-ft[2])*.6]);NJ[side+'_toetip']=np.array([ft[0],.03,zt])
        # braço em T ao longo de x: linha central, punho logo depois da munhequeira (faixa sem pele)
        xa=PB[:,0]*sg;arm=(xa>.2)&(PB[:,1]>NJ[side+'Arm'][1]-.25)&(PB[:,1]<NJ[side+'Arm'][1]+.2)
        tipx=xa[arm].max();xs=np.arange(.25,tipx,.01);sk_=[];cen=[]
        for x in xs:
            m=arm&(np.abs(xa-x)<.006)
            sk_.append(skin[m].mean() if m.sum()>5 else 1);cen.append(PB[m].mean(0) if m.sum()>5 else None)
        sk_=np.array(sk_);band=np.nonzero((sk_<.6)&(xs>.45*tipx))[0]
        xw=xs[band.max()]+.01 if len(band) else .77*tipx
        def cline(x):
            m=arm&(np.abs(xa-x)<.015);return PB[m].mean(0)
        S=NJ[side+'Arm'].copy();S[1:]=cline(max(S[0]*sg,.25))[1:]
        Wr=cline(xw);Wr[0]=xw*sg
        E=cline(S[0]*sg+(xw-S[0]*sg)*.486);E[0]=(S[0]*sg+(xw-S[0]*sg)*.486)*sg
        NJ[side+'Arm']=S;NJ[side+'ForeArm']=E;NJ[side+'Hand']=Wr
        # dedos: região além dos nós, agrupada em 4 por z; polegar = parte da frente perto do punho
        hand=arm&(xa>xw)
        hl=tipx-xw;xk=xw+hl*.55
        fz=PB[hand&(xa>xk+hl*.1)]
        zs=np.sort(fz[:,2]);zq=np.percentile(zs,[12.5,37.5,62.5,87.5])
        for it in range(20):
            lab=np.argmin(np.abs(fz[:,2][:,None]-zq[None,:]),1);zq=np.array([fz[lab==q,2].mean() for q in range(4)])
        order=np.argsort(-zq)  # índice na frente (+z) ... mínimo atrás
        for fi,f in enumerate(['Index','Middle','Ring','Pinky']):
            pts=fz[lab==order[fi]];tip=pts[np.argmax(pts[:,0]*sg)]
            base=np.array([xk*sg,pts[:,1].mean(),pts[:,2].mean()]);L=tip[0]*sg-xk
            for kk,fr in zip((1,2,3),(0,.4,.7)):NJ[f'{side}Hand{f}{kk}']=base+np.array([L*fr*sg,0,0])
            NJ[f'{side}Hand{f}_tip']=np.array([tip[0],base[1],base[2]])
        thumbm=hand&(xa<xk+hl*.15)&(PB[:,2]>np.percentile(PB[hand,2],80))
        tp=PB[thumbm];tt=tp[np.argmax(tp[:,0]*sg+tp[:,2])]
        t1=Wr+np.array([.02*sg,-.012,.02]);NJ[side+'HandThumb1']=t1
        NJ[side+'HandThumb2']=t1+(tt-t1)*.4;NJ[side+'HandThumb3']=t1+(tt-t1)*.72;NJ[side+'HandThumb_tip']=tt
        NJ[side+'HandMiddle1_']=NJ[side+'HandMiddle1']
        print(side,'ombro',S.round(3),'cotovelo',E.round(3),'punho',Wr.round(3),'ponta',round(tipx,3))
    NJ['Head_tip']=np.array([0,PB[:,1].max()-.03,NJ['Head'][2]])
    return NJ,k
def build_bind(NJ,k):
    """matrizes de ligação novas (no espaço antigo) e posições das juntas no espaço antigo"""
    toOld=lambda X:(np.asarray(X)/k+np.array([0,g0,0]))@FO.T+mF
    Rup={n:FO.T@BW[bi[n]][:3,:3] for n in names}
    Rnew={}
    def tipname(n):
        for t in [n+'_tip',n[:-1]+'_tip']:
            if t in NJ:return t
        return None
    order=['Hips']+[n for n in names if n!='Hips']
    for n in order:
        c=MAINCHILD.get(n,(CHILD.get(n) or [None])[0])
        tgt=None
        if c and c in NJ and not c.startswith('Tail'):tgt=c
        elif n=='Head':tgt='Head_tip'
        elif n.endswith('ToeBase'):tgt=n.replace('ToeBase','_toetip')
        elif n[-1]=='3' and 'Hand' in n:tgt=n[:-1]+'_tip'
        if tgt:
            do=(J0[bi[c]]-J0[bi[n]]) if c and tgt==c else None
            if do is None:
                # sem filho no esqueleto: direção antiga = a do pai até ele
                p_=PARENT.get(n);do=J0[bi[n]]-J0[bi[p_]]
                dn=NJ[tgt]-NJ[n]
            else:dn=NJ[tgt]-NJ[n]
            Rnew[n]=rot_between(do,dn)@Rup[n]
        else:
            p_=PARENT.get(n)
            Rnew[n]=(Rnew[p_]@Rup[p_].T@Rup[n]) if p_ in Rnew else Rup[n]
    IBn=IB.copy();JO={}
    for n in names:
        # rotação no espaço antigo (Rnew já carrega a escala das matrizes antigas)
        M=np.eye(4);M[:3,:3]=FO@Rnew[n]
        pos=toOld(NJ[n]);M[:3,3]=pos;JO[n]=pos
        IBn[bi[n]]=np.linalg.inv(M)
    return IBn,JO,toOld
def adjacency(X,T):
    q=np.round(X/1e-5).astype(np.int64);_,inv=np.unique(q,axis=0,return_inverse=True);inv=inv.ravel();Tw=inv[T];n=inv.max()+1
    return sp.coo_matrix((np.ones(Tw.size*2),(Tw[:,[0,1,2,1,2,0]].ravel(),Tw[:,[1,2,0,0,1,2]].ravel())),shape=(n,n)).tocsr(),inv,n
def segments(NJ,bones):
    segs=[]
    for n in bones:
        c=MAINCHILD.get(n,(CHILD.get(n) or [None])[0])
        if c and c in NJ and not c.startswith('Tail'):e=NJ[c]
        elif n=='Head':e=NJ['Head_tip']
        elif n.endswith('ToeBase'):e=NJ[n.replace('ToeBase','_toetip')]
        elif n[-1]=='3' and 'Hand' in n:e=NJ[n[:-1]+'_tip']
        else:e=NJ[n]+(NJ[n]-NJ[PARENT[n]])*.3
        segs.append((n,NJ[n],e))
    return segs
def skin_weights(X,T,segs,handMask,it=6):
    N=len(X);D=np.zeros((N,len(segs)))
    for k_,(n,a,e) in enumerate(segs):
        ab=e-a;L2=max(ab@ab,1e-12);t=np.clip(((X-a)@ab)/L2,0,1);D[:,k_]=nrm(X-(a+np.outer(t,ab)),axis=1)
    dmin=D.min(1,keepdims=True)
    Wt=np.where(D<dmin*1.35+.004,1/np.maximum(D,1e-3)**4,0);Wt/=Wt.sum(1,keepdims=True)
    A,inv,n=adjacency(X,T);dg=np.asarray(A.sum(1)).ravel()+1
    Wq=np.zeros((n,Wt.shape[1]));cn=np.zeros(n);np.add.at(Wq,inv,Wt);np.add.at(cn,inv,1);Wq/=cn[:,None]
    hm=np.zeros(n);np.add.at(hm,inv,handMask.astype(float));hm=(hm>0)
    for _ in range(it):
        S=(A@Wq+Wq)/dg[:,None];Wq=np.where(hm[:,None],Wq*.8+S*.2,S)
    Wt=Wq[inv]
    top=np.argsort(-Wt,axis=1)[:,:4];tw=np.take_along_axis(Wt,top,1);tw/=np.maximum(tw.sum(1,keepdims=True),1e-9)
    return top,tw
def write_glb(F,X,Nn,UV,T,JF,WF,IBn,src,extras={}):
    blobs=[];views=[];accs=[]
    def add(arr,typ,ct,target=None,minmax=False,normalized=False):
        off=sum(len(x) for x in blobs);bb=arr.tobytes();blobs.append(bb+b'\0'*((4-len(bb)%4)%4))
        vv={'buffer':0,'byteOffset':off,'byteLength':len(bb)}
        if target:vv['target']=target
        views.append(vv);a={'bufferView':len(views)-1,'componentType':ct,'count':len(arr),'type':typ}
        if normalized:a['normalized']=True
        if minmax:a['min']=arr.min(0).tolist();a['max']=arr.max(0).tolist()
        accs.append(a);return len(accs)-1
    iP=add(X.astype(np.float32),'VEC3',5126,34962,True);iN=add(Nn.astype(np.float32),'VEC3',5126,34962);iU=add(UV,'VEC2',5126,34962)
    iJ=add(JF.astype(np.uint8),'VEC4',5121,34962);iW=add(WF,'VEC4',5121,34962,normalized=True);iI=add(T.astype(np.uint32).ravel(),'SCALAR',5125,34963)
    images=[]
    for kk in range(len(src['j']['images'])):
        bb,mt=imgbytes(src['d'],src['j'],src['b0'],kk);off=sum(len(x) for x in blobs);blobs.append(bb+b'\0'*((4-len(bb)%4)%4));views.append({'buffer':0,'byteOffset':off,'byteLength':len(bb)})
        images.append({'bufferView':len(views)-1,'mimeType':mt})
    binb=b''.join(blobs)
    ex={'inverseBind':[float(x) for x in IBn.transpose(0,2,1).ravel()]};ex.update(extras)
    gl={'asset':{'version':'2.0','generator':'aethelred rig'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'malha','mesh':0}],
     'meshes':[{'name':'malha','primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU,'JOINTS_0':iJ,'WEIGHTS_0':iW},'indices':iI,'material':0}],'extras':ex}],
     'materials':[src['mat']],'textures':src['j']['textures'],'images':images,'accessors':accs,'bufferViews':views,'buffers':[{'byteLength':len(binb)}]}
    if src['j'].get('samplers'):gl['samplers']=src['j']['samplers']
    js=json.dumps(gl,separators=(',',':')).encode();js+=b' '*((4-len(js)%4)%4)
    out=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(binb))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(binb),0x004E4942)+binb
    open(F,'wb').write(out);print(F,len(out))
def wquant(top,tw):
    WF=np.clip(np.round(tw*255),0,255).astype(np.uint8);WF[:,0]+=(255-WF.astype(int).sum(1)).astype(np.uint8);return top.astype(np.uint8),WF
if __name__=="__main__":
    B=load_static(os.path.join(ROOT,'modelos','rogue_warrior_t_pose_leve.glb'))
    ti=B['j']['textures'][B['mat']['pbrMetallicRoughness']['baseColorTexture']['index']]['source']
    C=sample(np.asarray(Image.open(io.BytesIO(imgbytes(B['d'],B['j'],B['b0'],ti)[0])).convert('RGB')).astype(np.float64)/255,B['UV'])
    NJ,k=fit_joints(B,C)
    IBn,JO,toOld=build_bind(NJ,k)
    bones=[n for n in names if not n.startswith('Tail')]
    segs=segments(NJ,bones)
    handMask=np.zeros(len(B['P']),bool)
    for sg in (1,-1):handMask|=(B['P'][:,0]*sg>NJ['LeftHand'][0]*.98)
    top,tw=skin_weights(B['P'],B['I'],segs,handMask)
    top=np.array([bi[segs[t][0]] for t in top.ravel()]).reshape(top.shape)
    JF,WF=wquant(top,tw)
    X=toOld(B['P']);Nn=B['N']@FO.T
    np.save(os.path.join(ROOT,'ferramentas','.juntas_corpo.npy'),{'NJ':NJ,'k':k},allow_pickle=True)
    # vértices dos braços (escondidos quando os braços mutantes aparecem): mais da metade do peso no braço/antebraço/mão/dedos
    import base64
    armIds=[bi[n] for n in names if any(s_ in n for s_ in ['Arm','Hand']) and 'Shoulder' not in n]
    aw=(np.isin(top,armIds)*tw).sum(1)
    armHide=base64.b64encode(np.packbits((aw>.5).astype(np.uint8),bitorder='little').tobytes()).decode()
    write_glb(os.path.join(ROOT,'aethelred_corpo_rig.glb'),X,Nn,B['UV'],B['I'],JF,WF,IBn,B,{'armHide':armHide})
