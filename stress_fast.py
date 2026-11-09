import os
import sys,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"final_experiment"))
from train import load_data,features,rotate,logits,SEEDS,ARMS
_,_,test,y=load_data(root/"mnist.npz")
rng=np.random.default_rng(413)
noise=rng.normal(0,.15,test.shape).astype('f')
rows=[]
models={}
for arm in ARMS:
    for seed in SEEDS:
        with np.load(Path(os.environ.get("MNIST_OUTPUT",root))/"final_experiment"/"runs"/f"arm{arm:02}_seed{seed}.npz") as f:
            models[arm,seed]=[f[k] for k in ('w','b','v','c')]
def acc(z):
    return float(100*(z.argmax(1)==y).mean())
for angle in (0,-15,15,-30,30,-60,60):
    im=test if not angle else rotate(test,np.full(len(test),angle))
    x=features(im)
    sh=np.zeros_like(im); sh[:,:,3:]=im[:,:,:-3]
    shifted=features(sh)
    noisy=np.clip(x+noise.reshape(-1,784),0,1)
    views=[features(rotate(im,np.full(len(im),a))) for a in (-15,15)]
    for (arm,seed),p in models.items():
        raw=logits(x,p)
        ensemble=(raw+sum(logits(view,p) for view in views))/3
        rows.append(dict(arm=arm,seed=seed,angle=angle,clean=acc(raw),shift=acc(logits(shifted,p)),noise=acc(logits(noisy,p)),tta=acc(ensemble)))
(Path(os.environ.get("MNIST_OUTPUT",root))/"stress_fast.json").write_text(json.dumps(rows,indent=2))
