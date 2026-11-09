import os
from pathlib import Path
import json,sys
import numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"final_experiment"))
from train import load_data,split_labels,fit,rotate,features,logits,SEEDS
out=Path(os.environ.get("MNIST_OUTPUT", root))
im,y,test,target=load_data(root/"mnist.npz")
train,val=split_labels(y)
angles=(-45,-30,-15,0,15,30,45)
views=[features(test if not a else rotate(test,np.full(len(test),a))) for a in angles]
rows=[]
for n in (1000,5000):
    rng=np.random.default_rng(427)
    ids=np.concatenate([rng.permutation(train[y[train]==digit])[:n//10] for digit in range(10)])
    for seed in SEEDS:
        for arm in (0,15):
            params,history,seconds=fit(im,y,ids,val,seed,arm)
            name=f"limited_{n}_{arm}_{seed}"
            np.savez_compressed(out/(name+".npz"),**dict(zip(("w","b","v","c"),params,strict=True)))
            with np.load(out/(name+".npz")) as f:
                pred=np.stack([logits(view,[f[k] for k in ('w','b','v','c')]).argmax(1) for view in views])
            rows.append(dict(n=n,seed=seed,arm=arm,accuracy=(100*(pred==target).mean(1)).tolist(),history=history,seconds=seconds))
            (out/"limited.json").write_text(json.dumps(dict(angles=angles,rows=rows),indent=2))
