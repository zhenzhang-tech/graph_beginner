"""知识图谱补全训练与评估（TransE / RotatE / ConvE）。

用法:
    python train_kge.py --model TransE --epochs 50
    python train_kge.py --model RotatE --dim 200
"""
import argparse
import os
import random
import time

import torch
import torch.nn.functional as F

from kge_models import MODELS

DATA_DIR = r'D:/bysj/graph_data/FB15k-237'


def load_data():
    entity2id, rel2id = {}, {}
    data = {}

    def read(path):
        triples = []
        with open(path) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)
                triples.append((entity2id[h], rel2id[r], entity2id[t]))
        return triples

    for split in ['train', 'valid', 'test']:
        data[split] = read(os.path.join(DATA_DIR, f'{split}.txt'))

    return data, len(entity2id), len(rel2id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='TransE', choices=list(MODELS))
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--eval_subset', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data, num_entities, num_relations = load_data()
    train_triples = data['train']
    print(f'FB15k-237: {num_entities} entities, {num_relations} relations, '
          f'train {len(train_triples)}, valid {len(data["valid"])}, '
          f'test {len(data["test"])}')

    model = MODELS[args.model](num_entities, num_relations, dim=args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 正三元组集合（用于过滤评估）
    all_pos = set(train_triples) | set(data['valid']) | set(data['test'])
    hr2t, rt2h = {}, {}
    for (hh, rr, tt) in all_pos:
        hr2t.setdefault((hh, rr), set()).add(tt)
        rt2h.setdefault((rr, tt), set()).add(hh)
    train_tensor = torch.tensor(train_triples, dtype=torch.long, device=device)

    def sample_negatives(pos, num):
        h, r, t = pos[:, 0], pos[:, 1], pos[:, 2]
        corrupt_head = torch.randint(0, 2, (pos.size(0),), device=device).bool()
        neg_ent = torch.randint(0, num_entities, (pos.size(0),), device=device)
        neg_h = torch.where(corrupt_head, neg_ent, h)
        neg_t = torch.where(corrupt_head, t, neg_ent)
        return neg_h, r, neg_t

    def train_epoch():
        model.train()
        idx = torch.randperm(len(train_tensor), device=device)
        total_loss = 0.0
        n_batch = 0
        for i in range(0, len(idx), args.batch_size):
            batch_idx = idx[i:i + args.batch_size]
            pos = train_tensor[batch_idx]
            h, r, t = pos[:, 0], pos[:, 1], pos[:, 2]
            neg_h, neg_r, neg_t = sample_negatives(pos, num_entities)
            pos_score = model.score(h, r, t)
            neg_score = model.score(neg_h, neg_r, neg_t)
            loss = F.relu(args.margin - pos_score + neg_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        return total_loss / n_batch

    @torch.no_grad()
    def evaluate(triples):
        model.eval()
        ranks = []
        all_ent = torch.arange(num_entities, device=device)
        for i in range(0, len(triples), 8):
            batch = triples[i:i + 8]
            h = torch.tensor([x[0] for x in batch], device=device)
            r = torch.tensor([x[1] for x in batch], device=device)
            t = torch.tensor([x[2] for x in batch], device=device)
            B = h.size(0)

            # corrupt tail：对每个 (h, r)，对所有实体打分
            h_rep = h.repeat_interleave(num_entities)
            r_rep = r.repeat_interleave(num_entities)
            t_all = all_ent.repeat(B)
            tail_scores = model.score(h_rep, r_rep, t_all).view(B, num_entities)

            # corrupt head
            head_scores = model.score(
                all_ent.repeat(B), r_rep,
                t.repeat_interleave(num_entities)).view(B, num_entities)

            for j in range(B):
                hh, rr, tt = batch[j]
                for cand in hr2t.get((hh, rr), ()):
                    if cand != tt:
                        tail_scores[j, cand] = -float('inf')
                for cand in rt2h.get((rr, tt), ()):
                    if cand != hh:
                        head_scores[j, cand] = -float('inf')
            true_tail = tail_scores[torch.arange(B, device=device), t]
            true_head = head_scores[torch.arange(B, device=device), h]
            tr = (tail_scores > true_tail.unsqueeze(1)).sum(dim=1) + 1
            hr = (head_scores > true_head.unsqueeze(1)).sum(dim=1) + 1
            ranks.extend(zip(tr.cpu().tolist(), hr.cpu().tolist()))

        mrr = 0.0
        hits = {1: 0, 3: 0, 10: 0}
        for tr, hr in ranks:
            mrr += 1.0 / tr + 1.0 / hr
            for k in hits:
                hits[k] += int(tr <= k) + int(hr <= k)
        n = 2 * len(ranks)
        return mrr / n, {k: v / n for k, v in hits.items()}

    print(f'model={args.model} dim={args.dim} lr={args.lr} epochs={args.epochs}')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch()
        if epoch % 5 == 0 or epoch == 1:
            mrr, hits = evaluate(data['valid'][:args.eval_subset])
            print(f'Epoch {epoch:03d} | loss {loss:.4f} | val MRR {mrr:.4f} '
                  f'Hits@10 {hits[10]:.4f} | {time.time()-t0:.1f}s')

    mrr, hits = evaluate(data['test'][:args.eval_subset])
    print(f'RESULT test MRR={mrr:.4f} Hits@1={hits[1]:.4f} '
          f'Hits@3={hits[3]:.4f} Hits@10={hits[10]:.4f}')


if __name__ == '__main__':
    main()
