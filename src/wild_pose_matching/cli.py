from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(prog="gestureclr"); s=p.add_subparsers(dest="cmd",required=True)
    t=s.add_parser("train"); t.add_argument("--pairs",required=True); t.add_argument("--output",required=True); t.add_argument("--epochs",type=int,default=50); t.add_argument("--batch-size",type=int,default=64); t.add_argument("--seed",type=int,default=0)
    c=s.add_parser("cluster"); c.add_argument("--units",required=True); c.add_argument("--checkpoint",required=True); c.add_argument("--output",required=True); c.add_argument("--clusters",type=int,default=100)
    m=s.add_parser("mine"); m.add_argument("--wild",required=True); m.add_argument("--units",required=True); m.add_argument("--checkpoint",required=True); m.add_argument("--clusters",required=True); m.add_argument("--output",required=True); m.add_argument("--sbert",default="all-MiniLM-L6-v2")
    r=s.add_parser("retrieve"); r.add_argument("--rules",required=True); r.add_argument("--clusters",required=True); r.add_argument("--text",required=True); r.add_argument("--output",required=True); r.add_argument("--seed",type=int,default=0); r.add_argument("--sbert",default="all-MiniLM-L6-v2")
    a=p.parse_args()
    if a.cmd=="train": train(a)
    elif a.cmd=="cluster": cluster(a)
    elif a.cmd=="mine": mine(a)
    else: retrieve_cmd(a)

def _model(checkpoint,dim2,dim3):
    import torch
    from .model import GestureCLR
    model=GestureCLR(dim2,dim3); model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)); model.eval(); return model

def train(a):
    import torch
    from .model import GestureCLR,ntxent
    from .pipeline import augment_projected
    torch.manual_seed(a.seed); data=np.load(a.pairs); x2=data["pose2d"].astype("float32"); x3=data["motion3d"].astype("float32")
    if len(x2)!=len(x3) or len(x2)<2: raise ValueError("training requires at least two aligned 2D/3D pairs")
    if a.epochs<1 or a.batch_size<2: raise ValueError("epochs must be positive and batch size must be at least two")
    model=GestureCLR(x2.shape[-1],x3.shape[-1]); opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4); rng=np.random.default_rng(a.seed)
    for epoch in range(a.epochs):
        order=rng.permutation(len(x2)); total=0.; seen=0
        for start in range(0,len(order),a.batch_size):
            ids=order[start:start+a.batch_size]
            if len(ids)<2: continue
            aug=np.stack([augment_projected(x2[i],float(rng.choice([.001,.01,.1])),int(rng.integers(-15,16)),rng) for i in ids])
            z2,z3=model(torch.from_numpy(aug),torch.from_numpy(x3[ids])); loss=ntxent(z2,z3); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss.detach())*len(ids)
            seen+=len(ids)
        print(json.dumps({"epoch":epoch+1,"loss":total/seen}))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),a.output)

def cluster(a):
    import torch
    from .pipeline import cluster_latents
    d=np.load(a.units); x=d["motion3d"].astype("float32"); ids=[str(v) for v in d["ids"]]; model=_model(a.checkpoint,int(d["dim2"]),x.shape[-1])
    with torch.no_grad(): z=model.motion3d(torch.from_numpy(x)).numpy()
    labels,centers=cluster_latents(z,a.clusters); output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); np.savez(output,ids=np.asarray(ids),labels=labels,centroids=centers,latents=z)

def mine(a):
    import torch
    from sentence_transformers import SentenceTransformer
    from .pipeline import build_rules,read_jsonl,write_jsonl
    w=np.load(a.wild); u=np.load(a.units); cl=np.load(a.clusters); model=_model(a.checkpoint,w["pose2d"].shape[-1],u["motion3d"].shape[-1])
    with torch.no_grad(): wz=model.pose2d(torch.from_numpy(w["pose2d"].astype("float32"))).numpy(); uz=model.motion3d(torch.from_numpy(u["motion3d"].astype("float32"))).numpy()
    texts=[str(x) for x in w["texts"]]; emb=SentenceTransformer(a.sbert).encode(texts,normalize_embeddings=True)
    write_jsonl(a.output,build_rules(emb,texts,wz,uz,[str(x) for x in u["ids"]],cl["labels"]))

def retrieve_cmd(a):
    from sentence_transformers import SentenceTransformer
    from .pipeline import read_jsonl,retrieve
    rules=read_jsonl(a.rules); d=np.load(a.clusters); clusters={int(k):[str(x) for x in d["ids"][d["labels"]==k]] for k in np.unique(d["labels"])}; model=SentenceTransformer(a.sbert)
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(retrieve(a.text,rules,lambda x:model.encode(x,normalize_embeddings=True),clusters,a.seed),indent=2),encoding="utf-8")

if __name__=="__main__": main()
