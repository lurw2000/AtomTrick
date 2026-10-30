#!/usr/bin/env python3
"""Refined NVT conformation classification (Ac-Ala3-NHMe, 1 ns NVT at 300 K).
Per-residue backbone (phi,psi) sorted into 7 local classes + unassigned (priority order), with:
  - C7 H-bond flag for C7eq/invγ and C7ax/classic-γ: O(r-1)···H-N(r+1) < 2.5 A (the 7-membered
    γ-turn ring, centered on residue r).
  - C5-contact flag for β/C5: intra-residue N(r)-H···O=C(r) < 2.6 A (the 5-membered ring).
Backbone + H-bond donor/acceptor atoms auto-detected from connectivity (no hard-coded indices).

Inputs are the trajectory files written by md/md_openmm.py (<out>_traj.npz: z, positions,
stride_ps).  The first --window_ps of each trajectory is used; residues are pooled.

    python md/analysis/ramachandran_classes.py                    # reported 1 ns run
    python md/analysis/ramachandran_classes.py --traj ATOMTRICK=runs/md/a_nvt_traj.npz \
        --out ramachandran_classes.png

Without ``--traj`` the script reads the trajectory in ``md/results/reported_1ns/``.  Expected (reported run, 3 residues x 500
frames = 1500 samples): C7eq/invγ 23.6 %, αR 19.0, bridge/δR 16.1, PPII 1.3, β/C5 3.7, αL 0.1,
C7ax/γ 24.1, unassigned 12.2.
"""
import argparse
import os

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mpatches

_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
REPORTED = [("ATOMTRICK", os.path.join(_RES, "reported_1ns", "atomtrick_nvt_traj.npz"))]
TITLE = "NVT conformation classes — Ac-Ala3-NHMe (300 K, 1 ns, release driver, seed 2024)"
COV={1:0.31,6:0.76,7:0.71,8:0.66}
C7_CUT=2.5; C5_CUT=2.6

ap = argparse.ArgumentParser(description="Ramachandran class populations + H-bond flags")
ap.add_argument("--traj", action="append", default=None, metavar="LABEL=PATH",
                help="trajectory npz (repeatable); default: the reported run")
ap.add_argument("--window_ps", type=float, default=1000.0)
ap.add_argument("--out", default="ramachandran_classes.png")
ap.add_argument("--title", default=TITLE)
args = ap.parse_args()
ARMS = [tuple(s.split("=", 1)) for s in args.traj] if args.traj else REPORTED
WINDOW_PS = args.window_ps

def load(path):
    d=np.load(path)
    z=d["z"].astype(int); P=d["positions"].astype(float)
    st=float(d["stride_ps"]) if "stride_ps" in d else 2.0
    return z, P[:int(WINDOW_PS/st)]
def adjacency(pos,z):
    n=len(z); adj=[[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(pos[i]-pos[j])<1.3*(COV.get(z[i],.77)+COV.get(z[j],.77)):
                adj[i].append(j); adj[j].append(i)
    return adj
def residues(pos0,z):
    n=len(z); adj=adjacency(pos0,z)
    carbonyl={c for c in range(n) if z[c]==6 and any(z[k]==8 for k in adj[c])}
    out=[]
    for N in [i for i in range(n) if z[i]==7]:
        cs=[k for k in adj[N] if z[k]==6]; prevC=[c for c in cs if c in carbonyl]
        ca=[c for c in cs if c not in carbonyl and any(cc in carbonyl for cc in adj[c])]
        if not prevC or not ca: continue
        CA=ca[0]; Ci=[c for c in adj[CA] if c in carbonyl]
        if not Ci: continue
        nextN=[k for k in adj[Ci[0]] if z[k]==7 and k!=N]
        if not nextN: continue
        pC=prevC[0]; C=Ci[0]; nN=nextN[0]
        Op=[k for k in adj[pC] if z[k]==8]; Hn=[k for k in adj[nN] if z[k]==1]
        Hs=[k for k in adj[N] if z[k]==1]; Os=[k for k in adj[C] if z[k]==8]
        out.append(dict(pC=pC,N=N,CA=CA,C=C,nN=nN,
                        Op=Op[0] if Op else None, Hn=Hn[0] if Hn else None,
                        Hs=Hs[0] if Hs else None, Os=Os[0] if Os else None))
    out.sort(key=lambda r:r["N"]); return out
def dih(p0,p1,p2,p3):
    b0,b1,b2=p0-p1,p2-p1,p3-p2; b1=b1/np.linalg.norm(b1,axis=-1,keepdims=True)
    v=b0-(b0*b1).sum(-1,keepdims=True)*b1; w=b2-(b2*b1).sum(-1,keepdims=True)*b1
    return np.degrees(np.arctan2((np.cross(b1,v)*w).sum(-1),(v*w).sum(-1)))
def dist(P,a,b): return np.linalg.norm(P[:,a]-P[:,b],axis=-1)

# 7 classes + unassigned, assigned in PRIORITY order (first match wins)
LABELS=["C7eq/invγ","αR","bridge/δR","PPII","β/C5","αL","C7ax/γ","unassigned"]
BOXES=[  # (name, phi0,phi1, psi0,psi1) — priority order
    ("αR",        -125,-30,  -90,-10),
    ("C7eq/invγ", -125,-45,   40,100),
    ("PPII",      -100,-50,  105,181),
    ("β/C5",      -181,-100, 105,181),
    ("bridge/δR", -125,-30,  -10, 40),
    ("αL",          25,100,    5,100),
    ("C7ax/γ",      30,120, -110,-20),
]
def classify(phi,psi):
    lab=np.full(phi.shape,"unassigned",dtype=object)
    for name,f0,f1,s0,s1 in BOXES:
        m=(lab=="unassigned")&(phi>=f0)&(phi<f1)&(psi>=s0)&(psi<s1)
        lab[m]=name
    return lab

data={}
MODELS=[lab for lab,_ in ARMS]
for mdl,path in ARMS:
    z,P=load(path); res=residues(P[0],z)
    phi=np.concatenate([dih(P[:,r["pC"]],P[:,r["N"]],P[:,r["CA"]],P[:,r["C"]]) for r in res])
    psi=np.concatenate([dih(P[:,r["N"]],P[:,r["CA"]],P[:,r["C"]],P[:,r["nN"]]) for r in res])
    c7=np.concatenate([dist(P,r["Op"],r["Hn"]) if r["Op"] is not None and r["Hn"] is not None
                       else np.full(len(P),np.inf) for r in res])
    c5=np.concatenate([dist(P,r["Hs"],r["Os"]) if r["Hs"] is not None and r["Os"] is not None
                       else np.full(len(P),np.inf) for r in res])
    data[mdl]=dict(phi=phi,psi=psi,c7=c7,c5=c5,lab=classify(phi,psi),nres=len(res),nfr=len(P))

print("residues per model: " + ", ".join(f"{m} {data[m]['nres']}" for m in MODELS)
      + "; pooled samples: " + ", ".join(f"{m} {len(data[m]['phi'])}" for m in MODELS) + "\n")
print("| local class | " + " | ".join(f"{m} %" for m in MODELS) + " |")
print("|---|" + "---|"*len(MODELS))
for name in LABELS:
    row=[100*np.mean(data[m]["lab"]==name) for m in MODELS]
    print(f"| {name} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
print("\n| H-bond flag (within class) | " + " | ".join(MODELS) + " |")
print("|---|" + "---|"*len(MODELS))
def flag(mdl,cls,dkey,cut):
    d=data[mdl]; m=d["lab"]==cls
    return (100*np.mean(d[dkey][m]<cut)) if m.sum()>0 else float("nan")
for desc, cls, key, cut in ((f"C7 H-bond formed in C7eq/invγ (H···O<{C7_CUT}Å)", "C7eq/invγ", "c7", C7_CUT),
                            (f"C7 H-bond formed in C7ax/γ (H···O<{C7_CUT}Å)", "C7ax/γ", "c7", C7_CUT),
                            (f"C5 contact in β/C5 (N-H···O<{C5_CUT}Å)", "β/C5", "c5", C5_CUT)):
    print(f"| {desc} | " + " | ".join(f"{flag(m, cls, key, cut):.0f}%" for m in MODELS) + " |")

# figure: pooled density + class-region boxes
edges=np.arange(-180,181,6)
fig,axes=plt.subplots(1,len(MODELS),figsize=(6.5*len(MODELS),6.2),squeeze=False)
vmax=max(np.histogram2d(data[m]["phi"],data[m]["psi"],bins=[edges,edges],density=True)[0].max() for m in MODELS)
for ax,mdl in zip(axes[0],MODELS):
    H,_,_=np.histogram2d(data[mdl]["phi"],data[mdl]["psi"],bins=[edges,edges],density=True)
    ax.imshow(H.T,origin="lower",extent=[-180,180,-180,180],aspect="equal",cmap="magma",
              vmin=0,vmax=vmax)
    for name,f0,f1,s0,s1 in BOXES:
        ax.add_patch(mpatches.Rectangle((f0,s0),f1-f0,s1-s0,fill=False,ec="cyan",lw=1.0,alpha=0.8))
        ax.text((f0+f1)/2,(s0+s1)/2,name,color="cyan",fontsize=7,ha="center",va="center")
    ax.set_xlim(-180,180); ax.set_ylim(-180,180); ax.set_xticks([-180,-90,0,90,180]); ax.set_yticks([-180,-90,0,90,180])
    ax.set_xlabel("φ (deg)"); ax.set_ylabel("ψ (deg)"); ax.set_title(f"{mdl} — first {WINDOW_PS:g} ps (pooled residues)")
fig.suptitle(args.title)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.tight_layout(rect=[0,0,1,0.96]); outp=args.out; fig.savefig(outp,dpi=140)
print(f"\nsaved {outp}")
