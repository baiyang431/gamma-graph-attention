"""在同一公开图与固定划分上执行注意力/头数消融，不写入文件。"""
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from gat import GAT, add_self_loops

DATA = Path(__file__).resolve().parents[1] / "GNN 入门"
SEEDS = [42, 43, 44]
EPOCHS = 300
PATIENCE = 50


def macro_f1(pred, labels):
    counts = torch.bincount(labels * 3 + pred, minlength=9).reshape(3, 3)
    return (2 * counts.diag() / (counts.sum(0) + counts.sum(1)).clamp_min(1)).mean().item()


def load_data():
    rows = {}
    for name in ["nodes", "edges", "train"]:
        with (DATA / (name + ".csv")).open(encoding="utf-8-sig", newline="") as f:
            rows[name] = list(csv.DictReader(f))
    nodes = rows["nodes"]
    index = {row["id"]: i for i, row in enumerate(nodes)}
    if len(index) != len(nodes):
        raise ValueError("节点 ID 重复")
    x = torch.tensor([[float(row[k]) for k in ["f1","f2","f3","f4"]] for row in nodes])
    if not torch.isfinite(x).all():
        raise ValueError("特征包含非有限值")
    labels = torch.full((len(nodes),), -1, dtype=torch.long)
    train_ids = [row["id"] for row in rows["train"]]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("训练 ID 重复")
    for row in rows["train"]:
        label = int(row["target"])
        if label not in (0,1,2):
            raise ValueError("训练标签须为 0/1/2")
        labels[index[row["id"]]] = label
    train_idx = torch.tensor([index[node_id] for node_id in train_ids])
    if set(labels[train_idx].tolist()) != {0,1,2}:
        raise ValueError("训练标签须覆盖三类")
    edges = torch.tensor([[index[row["source"]],index[row["target"]]] for row in rows["edges"]]).t().contiguous()
    edges = add_self_loops(edges, len(nodes))
    generator = torch.Generator().manual_seed(42)
    splits = [[],[],[]]
    for label in range(3):
        group = train_idx[labels[train_idx] == label]
        group = group[torch.randperm(len(group), generator=generator)]
        size = round(len(group) * .2)
        for parts, values in zip(splits, [group[2*size:],group[size:2*size],group[:size]]):
            parts.extend(values.tolist())
    fit, val, holdout = [torch.tensor(part, dtype=torch.long) for part in splits]
    assert len(set(fit.tolist()+val.tolist()+holdout.tolist())) == len(train_idx)
    mean = x[fit].mean(0)
    scale = x[fit].std(0, unbiased=False).clamp_min(1e-8)
    x = (x - mean) / scale
    split_ids = {name: [nodes[i]["id"] for i in part.tolist()]
                 for name,part in zip(["fit","validation","holdout"],[fit,val,holdout])}
    metadata = {
        "nodes":len(nodes), "edges_with_loops":edges.shape[1],
        "split_counts":{name:len(ids) for name,ids in split_ids.items()},
        "split_sha256":hashlib.sha256(json.dumps(split_ids,sort_keys=True).encode()).hexdigest(),
        "data_sha256":{name:hashlib.sha256((DATA/(name+".csv")).read_bytes()).hexdigest()
                       for name in rows},
        "split_ids":split_ids,
    }
    return x, edges, labels, fit, val, holdout, metadata


class Network(nn.Module):
    def __init__(self, heads, attention):
        super().__init__()
        self.first = GAT(4,16//heads,heads,concat=True,attention=attention)
        self.second = GAT(16,3,1,concat=False,attention=attention)
        self.dropout = nn.Dropout(.2)

    def forward(self, x, edges):
        hidden = self.dropout(torch.nn.functional.elu(self.first(x,edges)))
        return self.second(hidden,edges)


def run():
    torch.set_num_threads(1)
    x, edges, labels, fit, val, holdout, metadata = load_data()
    print("DATA "+json.dumps(metadata,ensure_ascii=False),flush=True)
    loss_f = nn.CrossEntropyLoss()
    results = []
    for attention,heads in [(False,4),(True,1),(True,2),(True,4)]:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = Network(heads,attention)
            # 总隐藏宽度固定；各设置使用相同投影初值和 Dropout 随机流。
            torch.manual_seed(seed+1)
            with torch.no_grad():
                for layer in [model.first,model.second]:
                    nn.init.xavier_uniform_(layer.fc.weight)
            torch.manual_seed(seed+2)
            optimizer = torch.optim.Adam(model.parameters(),lr=.01,weight_decay=5e-4)
            best = (-1.,-float("inf"))
            state = None
            saved = stale = 0
            for epoch in range(1,EPOCHS+1):
                model.train()
                optimizer.zero_grad()
                loss = loss_f(model(x,edges)[fit],labels[fit])
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    outputs = model(x,edges)
                    val_loss = loss_f(outputs[val],labels[val]).item()
                    val_f1 = macro_f1(outputs[val].argmax(1),labels[val])
                current = (val_f1,-val_loss)
                if current > best:
                    best = current
                    state = deepcopy(model.state_dict())
                    saved = epoch
                    stale = 0
                else:
                    stale += 1
                if stale >= PATIENCE:
                    break
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                outputs = model(x,edges)
                row = {"attention":attention,"heads":heads,"seed":seed,
                       "parameters":sum(p.numel() for p in model.parameters()),
                       "best_epoch":saved,"run_epochs":epoch,
                       "train_loss":loss_f(outputs[fit],labels[fit]).item(),
                       "validation_loss":loss_f(outputs[val],labels[val]).item(),
                       "train_f1":macro_f1(outputs[fit].argmax(1),labels[fit]),
                       "validation_f1":macro_f1(outputs[val].argmax(1),labels[val]),
                       "holdout_f1":macro_f1(outputs[holdout].argmax(1),labels[holdout])}
            results.append(row)
            print("RUN "+json.dumps(row),flush=True)
    summaries = []
    for attention,heads in [(False,4),(True,1),(True,2),(True,4)]:
        group = [r for r in results if r["attention"]==attention and r["heads"]==heads]
        summary = {"attention":attention,"heads":heads,"parameters":group[0]["parameters"]}
        for key in ["train_f1","validation_f1","holdout_f1"]:
            values = torch.tensor([r[key] for r in group],dtype=torch.float64)
            summary[key+"_mean"] = values.mean().item()
            summary[key+"_std"] = values.std(unbiased=True).item()
        summaries.append(summary)
    print("SUMMARY "+json.dumps(summaries),flush=True)
    return {"metadata":metadata,"runs":results,"summaries":summaries}


if __name__ == "__main__":
    run()
