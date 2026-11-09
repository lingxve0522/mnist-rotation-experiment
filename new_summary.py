import os
from pathlib import Path
import json
import numpy as np
p=Path(os.environ.get("MNIST_OUTPUT",Path(__file__).parent))
original=p / "final_experiment"
def read(name,root=p): return json.loads((root/(name+".json")).read_text(encoding="utf-8"))
s=read("summary",original)
L=read("linear"); R=read("limited"); T=read("stress_fast")
assert len(L)==15 and len(R["rows"])==20 and len(T)==140
angles=(-45,-30,-20,-15,-10,-5,0,5,10,15,20,30,45)
def arr(rows,a): return np.array([r['accuracy'] for r in sorted(rows,key=lambda r:r['seed']) if r['arm']==a])
def sel(a,idx,ang): return a[:,[ang.index(i) for i in idx]].mean(1)
ends=((0,),(-15,15),(-30,30),(-45,45))
fmt=lambda v:f"{np.mean(v):.2f} ± {np.std(v,ddof=1):.2f}"
lin=[]
for arm in (0,15,30):
    a=arr(L,arm)
    lin.append([str(arm)]+[fmt(sel(a,k,angles)) for k in ends])
small=[]
for n in (1000,5000):
    rr=[r for r in R['rows'] if r['n']==n]
    for arm in (0,15):
        small.append([str(n),str(arm)]+[fmt(sel(arr(rr,arm),k,R['angles'])) for k in ends])
stress=[]
for arm in (0,5,15,30):
    for pair in ((0,),(-15,15),(-30,30),(-60,60)):
        a=np.array([[np.mean([r[key] for r in T if r['arm']==arm and r['seed']==seed and r['angle'] in pair]) for key in ('clean','shift','noise','tta')] for seed in range(123,128)])
        stress.append([str(arm),str(max(pair))]+[fmt(a[:,i]) for i in range(4)])
angles_full=[[str(a)]+[fmt(np.array(s['arms'][str(m)]['by_angle'][str(a)]['values'])) for m in (0,5,15,30)] for a in angles]
linear_full=[[str(a)]+[fmt(arr(L,m)[:,j]) for m in (0,15,30)] for j,a in enumerate(angles)]
deltas=[]
for name,rr,ang,arm in [('线性模型',L,angles,15)]+[(f'MLP {n} 张',[r for r in R['rows'] if r['n']==n],R['angles'],15) for n in (1000,5000)]:
    for pair in ends:
        d=sel(arr(rr,arm)-arr(rr,0),pair,ang)
        half=2.776445*np.std(d,ddof=1)/np.sqrt(5)
        deltas.append([name,str(max(pair)),f"{d.mean():+.2f}",f"[{d.mean()-half:+.2f}, {d.mean()+half:+.2f}]"])
data=dict(linear=lin,limited=small,stress=stress,angles_full=angles_full,linear_full=linear_full,paired=deltas)
(p/"new_summary.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
for k in ('linear','limited','stress','paired'):
    print(k,data[k])
