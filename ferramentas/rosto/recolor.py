import numpy as np; from PIL import Image, ImageFilter
t=np.asarray(Image.open('tex.png').convert('RGB'),np.float32); lab=np.load('lab.npy')
# espalha os rótulos para as bordas (gutters) para não sobrar contorno cinza
L=Image.fromarray(lab)
for _ in range(4): 
    d=np.asarray(L.filter(ImageFilter.MaxFilter(5))); lab=np.where(lab==0,d,lab); L=Image.fromarray(lab)
mx=t.max(2); r,g,b=t[...,0],t[...,1],t[...,2]
rg=(r-np.maximum(g,b))/(mx+10)
neutral=np.clip((0.17-rg)/0.09,0,1)          # cinza/preto da armadura; capa, pele e braços (avermelhados) ficam fora
PAL={ # escuro, médio, claro (tirado da foto de referência)
 1:[(16,11,9),(48,33,24),(84,62,46)],    # capuz marrom escuro
 2:[(28,14,12),(80,38,30),(128,72,56)],   # colete / ombros: couro marrom avermelhado
 3:[(26,14,12),(74,36,28),(118,66,50)],    # saia / coxas
 4:[(22,19,16),(54,44,34),(92,78,62)],    # calça marrom escura
 5:[(24,16,11),(66,44,30),(108,76,54)],   # botas
 6:[(28,14,12),(80,38,30),(128,72,56)],   # peças cinzas nos braços (braçadeiras)
}
out=t.copy()
for k,(d,m_,l) in PAL.items():
    msk=lab==k
    if not msk.any(): continue
    ref=np.percentile(mx[msk&(neutral>.5)],95) if (msk&(neutral>.5)).any() else 80
    v=np.clip(mx/ref,0,1.4)
    d,m_,l=map(np.array,(d,m_,l))
    c=np.where((v<.6)[...,None], d+(m_-d)*(v/.6)[...,None], m_+(l-m_)*((v-.6)/.5)[...,None])
    w=neutral*msk
    if k==6: w=w*np.clip((95-mx)/35,0,1)   # garras e ossos claros ficam como estão
    w=w[...,None]
    out=out*(1-w)+c*w
Image.fromarray(out.clip(0,255).astype(np.uint8)).save('tex_rec.png')
