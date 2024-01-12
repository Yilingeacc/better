import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from config import Config
from data import DataLoader

item_num = 3953
emb_size = 32
sample_size = 200

movie_list = [{}]
tag_dict = {}
with open('../data/MovieLens-Rand/movies.dat', 'r', encoding='ISO-8859-1') as f:
    lines = f.read().split('\n')
    index = 1
    for line in lines:
        id, metadata, tag = line.split('::')
        if index != int(id):
            movie_list.append({})
            index += 1
        name = metadata[:metadata.rfind('(') - 1]
        year = int(metadata[metadata.rfind('(') + 1:metadata.rfind(')')])
        tags = tag.split('|')
        for tag in tags:
            if tag not in tag_dict:
                tag_dict[tag] = 1
            else:
                tag_dict[tag] += 1
        movie_list.append({'name': name, 'year': year, 'tags': tags})
        index += 1

tag_name2id = {}
for i, tag in enumerate(tag_dict.keys()):
    tag_name2id[tag] = i

tag_id2name = {}
for tag, id in tag_name2id.items():
    tag_id2name[id] = tag

for i, movie in enumerate(movie_list):
    if movie:
        tags = movie['tags']
        chosen = tags[0]
        cnt = tag_dict[chosen]
        for tag in tags:
            if tag_dict[tag] > cnt:
                chosen = tag
                cnt = tag_dict[tag]
        movie['label'] = tag_name2id[chosen]


def build_mlp(dims: [int]) -> nn.Sequential:
    net_list = list()
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]
    return nn.Sequential(*net_list)


class Model(nn.Module):
    def __init__(self, config, group2members):
        super().__init__()
        self.user_embedding = nn.Embedding(config.user_num + 1, config.embedding_size).to(config.device)
        self.item_embedding = nn.Embedding(config.item_num + 1, config.embedding_size).to(config.device)
        self.tag_embedding = nn.Embedding(config.tag_num + 1, config.embedding_size).to(config.device)
        self.group_dat_embedding = nn.Embedding(config.total_group_num + 1, config.embedding_size).to(config.device)
        self.item_dat_embedding = nn.Embedding(config.item_num + 1, config.embedding_size).to(config.device)

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

    def item(self, item_id):
        tags_id = self.item2tags[item_id]
        item_id_tensor = torch.tensor(item_id, dtype=torch.int).to(self.device)
        tags_id_tensor = torch.tensor(tags_id, dtype=torch.int).to(self.device)
        item_embedding = self.item_embedding(item_id_tensor)
        item_dat_embedding = self.item_dat_embedding(item_id_tensor).to(self.device)
        tags_embedding = self.tag_embedding(tags_id_tensor).mean(dim=0).squeeze()
        tower_input = torch.cat([item_embedding, tags_embedding, torch.zeros_like(item_dat_embedding)])
        return self.item_tower(tower_input), item_dat_embedding

    def items(self, item_ids):
        res = [self.item(id) for id in item_ids]
        result = torch.stack([r[0] for r in res])
        dat_embs = torch.stack([r[1] for r in res])
        return result, dat_embs


config = Config(dataset='MovieLens-Rand')
config.device = torch.device('cpu')
data_loader = DataLoader(config)
model = Model(config=config, group2members=data_loader.group2members_dict)


rating_matrix_pred = np.load('saves/MovieLens-Rand/env_MovieLens-Rand_32.npy')
available_item = [i for i in range(1, item_num) if any(rating_matrix_pred[:, i] > 0)]

tsne = TSNE(n_components=2)

colors = []
for i in range(9):
    colors.append(plt.cm.Set1(i))
for i in range(8):
    colors.append(plt.cm.Set2(i))
for i in range(12):
    colors.append(plt.cm.Set3(i))

test_no = 'test'

model.load_state_dict(torch.load(f'./model/MovieLens-Rand/better_{test_no}.pth', map_location=torch.device('cpu')), strict=False)
embeds = model.items(available_item)[0].detach().cpu().numpy()
# index = np.random.choice(available_item, sample_size, replace=False)
# sample = embeds[available_item]
X_tsne = tsne.fit_transform(embeds)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))

for i, id in enumerate(available_item):
    if movie_list[id]:
        label = movie_list[id]['label']
        plt.text(X_norm[i, 0], X_norm[i, 1], '·', color=colors[label],
                 fontdict={'weight': 'bold', 'size': 20})

plt.xticks([])
plt.yticks([])
plt.savefig(f'../fig/{test_no}.png')
