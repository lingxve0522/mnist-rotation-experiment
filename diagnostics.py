import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("--work",type=Path,required=True)
p.add_argument("--data",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
exp=a.work/"final_experiment"
sys.path.insert(0,str(Path(__file__).parent/"final_experiment"))
from train import load_data,rotate,features,logits,ARMS,SEEDS
_,_,im,y=load_data(a.data)
truth=np.eye(10)[y]
models={}
for arm in ARMS:
 for seed in SEEDS:
    with np.load(exp/"runs"/f"arm{arm:02}_seed{seed}.npz") as m:
        models[arm,seed]=[m[k] for k in ("w","b","v","c")]
rows=[]
transitions=[]
for angle in (-45,-30,-15,0,15,30,45):
    x=features(im if angle==0 else rotate(im,np.full(len(im),angle)))
    baseline={}
    for (arm,seed),params in models.items():
        z=logits(x,params).astype("float64")
        z-=z.max(1,keepdims=True)
        logp=z-np.log(np.exp(z).sum(1,keepdims=True))
        prob=np.exp(logp); pred=prob.argmax(1); conf=prob.max(1); correct=pred==y
        bins=np.minimum((conf*15).astype(int),14)
        counts=np.bincount(bins,minlength=15)
        gap=np.bincount(bins,weights=conf-correct,minlength=15)
        ece=float(np.abs(gap).sum()/len(y))
        cm=np.bincount(10*y.astype(int)+pred,minlength=100).reshape(10,10)
        rows.append(dict(arm=arm,seed=seed,angle=angle,acc=float(correct.mean()),confidence=float(conf.mean()),ece=ece,nll=float(-logp[np.arange(len(y)),y].mean()),brier=float(np.square(prob-truth).sum(1).mean()),confusion=cm.tolist()))
        if arm==0: baseline[seed]=correct
        else:
            previous=baseline[seed]
            transitions.append(dict(arm=arm,seed=seed,angle=angle,corrected=np.bincount(y[correct&~previous],minlength=10).tolist(),regressed=np.bincount(y[previous&~correct],minlength=10).tolist()))
assert len(rows)==140 and len(transitions)==105
a.out.write_text(json.dumps(dict(rows=rows,transitions=transitions),indent=2),encoding="utf-8")
