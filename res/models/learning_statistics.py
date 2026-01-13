import re, json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _f(x):
    try:
        v=float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def _ma(y,w):
    y=np.asarray(y,dtype=float)
    if y.size==0 or w<=1:
        return y
    k=np.ones(int(w),dtype=float)/float(w)
    m=np.convolve(np.nan_to_num(y,nan=0.0),k,mode="same")
    c=np.convolve(np.isfinite(y).astype(float),k,mode="same")
    return np.where(c>1e-9,m/np.maximum(c,1e-9),np.nan)

def _runmean(y):
    y=np.asarray(y,dtype=float)
    m=np.isfinite(y)
    s=np.cumsum(np.where(m,y,0.0))
    c=np.cumsum(m.astype(float))
    return np.where(c>0.0,s/np.maximum(c,1e-9),np.nan)

def _parse_log(txt):
    upd=[]; ep=[]
    rx_u=re.compile(r"\[PPO update #(\d+)\]\s+steps=(\d+)\s+avg_r/transition=([-\d\.eE+naninf]+)")
    for line in txt.splitlines():
        m=rx_u.search(line)
        if m:
            upd.append((int(m.group(1)),int(m.group(2)),_f(m.group(3))))
            continue
        if "Episode ended" in line and "duration_ticks=" in line:
            m2=re.search(r"duration_ticks=(\d+)", line)
            if m2:
                ep.append((len(ep)+1,int(m2.group(1))))
    return upd,ep

def _parse_json(path):
    try:
        obj=json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [],[]
    upd=[]; ep=[]
    if isinstance(obj,list):
        for it in obj:
            if not isinstance(it,dict):
                continue
            if "avg_reward_per_transition" in it or "avg_reward" in it:
                upd.append((int(it.get("update",len(upd)+1)),int(it.get("steps",0)),_f(it.get("avg_reward_per_transition",it.get("avg_reward",np.nan)))))
            if "duration_ticks" in it:
                try:
                    ep.append((len(ep)+1,int(it.get("duration_ticks",0))))
                except Exception:
                    pass
    return upd,ep

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--log",default="")
    ap.add_argument("--json",default="")
    ap.add_argument("--out",default="")
    ap.add_argument("--reward_ma",type=int,default=15)
    args=ap.parse_args()

    here=Path(__file__).resolve().parent
    jet_dir=here/"jet_ai"
    log_path=Path(args.log) if args.log else (jet_dir/"train.log")
    json_path=Path(args.json) if args.json else (jet_dir/"train_stats.json")
    out_path=Path(args.out) if args.out else (jet_dir/"ai_stats.png")

    upd=[]; ep=[]
    if json_path.exists():
        u,e=_parse_json(json_path); upd+=u; ep+=e
    if log_path.exists():
        u,e=_parse_log(log_path.read_text(encoding="utf-8",errors="ignore")); upd+=u; ep+=e
    if not upd and not ep:
        print("No stats found. Need jet_ai/train.log or jet_ai/training_stats.json")
        return

    if upd:
        upd=sorted(upd,key=lambda x:x[0])
        u=np.array(upd,dtype=float)
        ux=u[:,0]; ur=u[:,2]
        ur_ma=_ma(ur,max(1,args.reward_ma))
    else:
        ux=ur=ur_ma=np.array([],dtype=float)

    if ep:
        et=np.array([t for _,t in ep],dtype=float)
        ex=np.arange(1,et.size+1,dtype=float)
        et_mean=_runmean(et)
    else:
        ex=et=et_mean=np.array([],dtype=float)

    fig=plt.figure(figsize=(12.5,6.3))
    ax=fig.add_subplot(1,1,1)
    ax.set_title("Training: avg_r/transition and survival time")
    ax.set_xlabel("PPO update (green) / Episode (red)")
    ax.set_ylabel("avg_r/transition")
    if ux.size:
        ax.plot(ux,ur,alpha=0.18,color="green")
        ax.plot(ux,ur_ma,linewidth=2.8,color="green",label=f"avg_r/transition (MA {max(1,args.reward_ma)})")
    else:
        ax.text(0.5,0.65,"No PPO update lines found",ha="center",va="center",transform=ax.transAxes)

    ax2=ax.twinx()
    ax2.set_ylabel("Survival time (ticks)")
    if ex.size:
        ax2.scatter(ex,et,s=18,alpha=0.8,color="red",label="survival (ticks)",zorder=3)
        ax2.plot(ex,et_mean,linewidth=2.2,color="red",alpha=0.4,label="survival running mean")
    else:
        ax.text(0.5,0.35,"No episode-end durations found",ha="center",va="center",transform=ax.transAxes)

    ax.grid(True,alpha=0.2)

    h1,l1=ax.get_legend_handles_labels()
    h2,l2=ax2.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1+h2,l1+l2,loc="upper left",framealpha=0.9)

    note=[]
    if ux.size and np.isfinite(ur_ma).any():
        note.append(f"reward MA(last): {ur_ma[-1]:.4f}")
    if ex.size and np.isfinite(et_mean).any():
        note.append(f"survival mean(last): {et_mean[-1]:.1f} ticks")
    if note:
        ax.text(0.02,0.02," | ".join(note),transform=ax.transAxes,fontsize=10,alpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out_path,dpi=180)
    print(str(out_path))
    plt.show()

if __name__=="__main__":
    main()
