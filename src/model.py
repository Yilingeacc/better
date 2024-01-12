from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(dims: [int]) -> nn.Sequential:
    net_list = list()
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]
    return nn.Sequential(*net_list)


def clones(module: nn.Module, n: int):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Implementation
    See: Attention is all you need(https://arxiv.org/abs/1706.03762)
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        d_model = d_model
        d_k = d_model // n_head
        assert d_k * n_head == d_model
        self.h = n_head
        self.d_k = d_k
        self.d_model = d_model
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value):
        q, k, v = [l(x).view(-1, self.h, self.d_k).transpose(0, 1) for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5
        attn = F.softmax(scores, dim=-1)
        x = torch.matmul(attn, v).transpose(0, 1).contiguous().view(-1, self.h * self.d_k)
        return self.linears[-1](x)


class Model(nn.Module):
    def __init__(self, config, group2members):
        super().__init__()
        self.user_embedding = nn.Embedding(config.user_num + 1, config.embedding_size).to(config.device)
        self.item_embedding = nn.Embedding(config.item_num + 1, config.embedding_size).to(config.device)
        self.tag_embedding = nn.Embedding(config.tag_num + 1, config.embedding_size).to(config.device)
        self.group_dat_embedding = nn.Embedding(config.total_group_num + 1, config.embedding_size).to(config.device)
        self.item_dat_embedding = nn.Embedding(config.item_num + 1, config.embedding_size).to(config.device)

        # self.self_attn = MultiHeadAttention(config.embedding_size, config.nhead).to(config.device)
        # self.cross_attn = MultiHeadAttention(config.embedding_size, config.nhead).to(config.device)
        # self.proj = nn.Linear(config.embedding_size, 1).to(config.device)

        self.user_tower = build_mlp([config.user_tower_input_size, config.user_tower_hidden_size, config.embedding_size]).to(config.device)
        self.item_tower = build_mlp([config.item_tower_input_size, config.item_tower_hidden_size, config.embedding_size]).to(config.device)
        self.embedding_size = config.embedding_size

        self.group2members = group2members
        self.item2tags = {}
        self.device = config.device

        with open('id2typeid.txt', 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                id, tag = line.split(':')
                self.item2tags[int(id)] = [int(id) for id in tag.split(',')]

    def user(self, group_id):
        member_id = self.group2members[group_id]
        group_id_tensor = torch.tensor(group_id, dtype=torch.int).to(self.device)
        member_id_tensor = torch.tensor(member_id, dtype=torch.int).to(self.device)
        user_embedding = self.user_embedding(member_id_tensor).mean(dim=0)
        group_dat_embedding = self.group_dat_embedding(group_id_tensor)
        # group_embedding = self.self_attn(user_embedding, user_embedding, user_embedding).mean(dim=0)
        return self.user_tower(torch.cat([user_embedding, group_dat_embedding])), group_dat_embedding

    def item(self, item_id):
        if item_id not in self.item2tags:
            return torch.zeros(self.embedding_size), torch.zeros(self.embedding_size), torch.zeros(self.embedding_size), torch.zeros(self.embedding_size)
        tags_id = self.item2tags[item_id]
        item_id_tensor = torch.tensor(item_id, dtype=torch.int).to(self.device)
        tags_id_tensor = torch.tensor(tags_id, dtype=torch.int).to(self.device)
        item_embedding = self.item_embedding(item_id_tensor)
        tags_embedding = self.tag_embedding(tags_id_tensor).mean(dim=0).squeeze()
        item_dat_embedding = self.item_dat_embedding(item_id_tensor).to(self.device)
        tower_input = torch.cat([item_embedding, tags_embedding, item_dat_embedding])
        only_id = self.item_tower(torch.cat([item_embedding, torch.zeros_like(tags_embedding), item_dat_embedding]))
        only_tag = self.item_tower(torch.cat([torch.zeros_like(item_embedding), tags_embedding, item_dat_embedding]))
        return self.item_tower(tower_input), item_dat_embedding, only_id, only_tag

    def users(self, group_ids):
        res = [self.user(id) for id in group_ids]
        result = F.normalize(torch.stack([r[0] for r in res]), p=2, dim=1)
        dat_embs = F.normalize(torch.stack([r[1] for r in res]), p=2, dim=1)
        return result, dat_embs

    def items(self, item_ids):
        res = [self.item(id) for id in item_ids]
        result = F.normalize(torch.stack([r[0] for r in res]), p=2, dim=1)
        dat_embs = F.normalize(torch.stack([r[1] for r in res]), p=2, dim=1)
        only_id = F.normalize(torch.stack([r[2] for r in res]), p=2, dim=1)
        only_tag = F.normalize(torch.stack([r[3] for r in res]), p=2, dim=1)
        label = torch.tensor([self.labels[id] for id in item_ids], dtype=torch.long)
        return result, dat_embs, only_id, only_tag, label

    def save(self, e):
        print(f'save to disk: ./model/MovieLens-Rand/{e}.pth')
        torch.save(self.state_dict(), f'./model/MovieLens-Rand/{e}.pth')
