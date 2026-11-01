import numpy as np, json, sys
from PIL import Image
from scipy.spatial import Delaunay
# pontos na arte (concept_front.png, 425x665) -> no render ortogonal do modelo (800x800)
P={
 'olhoE':((170,267.5),(305,438)),'olhoD':((295,268.5),(495,438)),
 'cantoEext':((145,267.5),(265,437)),'cantoEint':((200,270),(340,442)),
 'cantoDint':((270,270),(455,442)),'cantoDext':((322.5,267.5),(535,437)),
 'sobrEint':((202.5,237.5),(350,395)),'sobrDint':((260,237.5),(450,395)),
 'sobrEext':((137.5,245),(252,402)),'sobrDext':((325,242.5),(552,400)),
 'narizPonta':((232.5,335),(400,545)),'narizBase':((232.5,350),(400,562)),
 'asaE':((205,345),(355,550)),'asaD':((262.5,345),(445,550)),
 'bocaE':((182.5,399),(335,625)),'bocaD':((277.5,399),(465,625)),
 'bocaC':((230,400),(400,628)),'labioSup':((230,382.5),(400,605)),'labioInf':((230,420),(400,660)),
 'queixo':((230,470),(400,750)),
}
src=np.array([v[0] for v in P.values()]); dst=np.array([v[1] for v in P.values()])
# similaridade global (mínimos quadrados) para pontos de contorno
def sim(a,b):
    A=[];B=[]
    for (x,y),(u,v) in zip(a,b): A+= [[x,-y,1,0],[y,x,0,1]]; B+=[u,v]
    p=np.linalg.lstsq(np.array(A,float),np.array(B,float),rcond=None)[0]
    return lambda q: np.stack([p[0]*q[:,0]-p[1]*q[:,1]+p[2], p[1]*q[:,0]+p[0]*q[:,1]+p[3]],1)
f=sim(src,dst); print('erro sim',np.abs(f(src)-dst).mean())
# contorno: anel de pontos ao redor do rosto na arte, mapeados pela similaridade
ang=np.linspace(0,2*np.pi,16,endpoint=False); cx,cy=230,330
ring=np.stack([cx+150*np.cos(ang),cy+190*np.sin(ang)],1)
ring2=np.stack([cx+600*np.cos(ang),cy+700*np.sin(ang)],1)
S=np.vstack([src,ring,ring2]); D=np.vstack([dst,f(ring),f(ring2)])
tri=Delaunay(D)
# para cada pixel do destino (800x800), acha triângulo e coordenada baricêntrica -> pixel na arte
H=W=800; yy,xx=np.mgrid[0:H,0:W]; pts=np.stack([xx.ravel()+.5,yy.ravel()+.5],1)
s=tri.find_simplex(pts); T=tri.transform[s]
b=np.einsum('nij,nj->ni',T[:,:2],pts-T[:,2]); bc=np.c_[b,1-b.sum(1)]
idx=tri.simplices[s]; sp=np.einsum('ni,nij->nj',bc,S[idx])
im=np.asarray(Image.open('meshy/concept_front.png').convert('RGB')).astype(np.float32)
# amostragem bilinear
x=np.clip(sp[:,0]-.5,0,im.shape[1]-1.001); y=np.clip(sp[:,1]-.5,0,im.shape[0]-1.001)
x0=x.astype(int); y0=y.astype(int); fx=(x-x0)[:,None]; fy=(y-y0)[:,None]
c=(im[y0,x0]*(1-fx)*(1-fy)+im[y0,x0+1]*fx*(1-fy)+im[y0+1,x0]*(1-fx)*fy+im[y0+1,x0+1]*fx*fy)
c[s<0]=0
Image.fromarray(c.reshape(H,W,3).clip(0,255).astype(np.uint8)).save('meshy/concept_warp.png')
