import json,struct,numpy as np
def load(p):
    b=open(p,'rb').read()
    jl=struct.unpack('<I',b[12:16])[0]; j=json.loads(b[20:20+jl])
    bin_=b[20+jl+8:]
    return j,bin_
CT={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}
NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
def acc(j,bin_,i):
    a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]
    off=bv.get('byteOffset',0)+a.get('byteOffset',0); n=NC[a['type']]; dt=CT[a['componentType']]
    stride=bv.get('byteStride',0); isz=np.dtype(dt).itemsize*n
    if stride and stride!=isz:
        raw=np.frombuffer(bin_,np.uint8,a['count']*stride,off).reshape(-1,stride)[:,:isz]
        return raw.copy().view(dt).reshape(a['count'],n)
    return np.frombuffer(bin_,dt,a['count']*n,off).reshape(a['count'],n)
