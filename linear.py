import os
import sys,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"final_experiment"))
from train import load_data,split_labels,features,rotate,ANGLES,SEEDS
out=Path(os.environ.get("MNIST_OUTPUT", root))
im,y,test,t=load_data(root/"mnist.npz")
ids,val=split_labels(y)
xs=[features(test if a==0 else rotate(test,np.full(len(test),a))) for a in ANGLES]
rows=[]
for seed in SEEDS:
    for arm in (0,15,30):
        rng=np.random.default_rng(seed)
        order=np.random.default_rng(seed+1000)
        aug=np.random.default_rng(seed+2000)
        w=rng.normal(0,.01,(784,10)).astype('f'); b=np.zeros(10,'f')
        vw=np.zeros_like(w); vb=np.zeros_like(b)
        history=[]
        for ep in range(15):
            lr=.005+.045*(1+np.cos(np.pi*ep/14))/2
            shuffled=order.permutation(ids)
            for offset in range(0,len(ids),256):
                j=shuffled[offset:offset+256]
                x=features(rotate(im[j],aug.uniform(-arm,arm,len(j))) if arm else im[j])
                p=x@w+b; p=np.exp(p-p.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
                p[np.arange(len(j)),y[j]]-=1; p/=len(j)
                vw=.9*vw+x.T@p; vb=.9*vb+p.sum(0); w-=lr*vw; b-=lr*vb
            history.append(float(((features(im[val])@w+b).argmax(1)==y[val]).mean()))
        name=f"linear_{arm}_{seed}"
        np.savez_compressed(out/(name+".npz"),w=w,b=b)
        with np.load(out/(name+".npz")) as m:
            pred=np.stack([(x@m['w']+m['b']).argmax(1) for x in xs]).astype('uint8')
        np.save(out/(name+".npy"),pred)
        rows.append(dict(seed=seed,arm=arm,accuracy=(100*(pred==t).mean(1)).tolist(),validation=history))
        (out/"linear.json").write_text(json.dumps(rows,indent=2))
        print(name, rows[-1]['accuracy'],flush=True)

