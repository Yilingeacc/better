import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from data import DataLoader
from dataset import Dataset
from eval import Evaluator
from model import Model


def info_nce_loss(sample, emb_list, label, temp):
    # Normalize the input embeddings

    sample = F.normalize(sample, dim=1)
    emb_list = F.normalize(emb_list, dim=2)

    batch_size, list_num, emb_size = emb_list.size()
    pos_indices = label.view(-1)  # Flatten the label tensor
    pos_samples = emb_list[torch.arange(batch_size), pos_indices]

    pos_scores = torch.sum(sample * pos_samples, dim=1)
    neg_samples = emb_list.view(batch_size * list_num, emb_size)
    neg_scores = torch.matmul(sample, neg_samples.t())
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    logits /= temp
    log_probs = F.log_softmax(logits, dim=1)
    loss = -log_probs[:, 0].mean()
    return loss


def train(config: Config, model: Model, dataset: Dataset):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()
    for i in range(config.epoch):
        batch = dataset.sample(config.batch_size)
        batch_group_embs, batch_group_dat_embs = model.users(batch[:, 0])
        batch_pos_embs, batch_item_dat_embs, pos_only_id, pos_only_tag, pos_label = model.items(batch[:, 1])
        batch_neg_embs, _, neg_only_id, neg_only_tag, neg_label = model.items(batch[:, 2])

        xuij = torch.mul(batch_group_embs, batch_pos_embs - batch_neg_embs).sum(dim=1, keepdim=True)
        main_loss = -torch.log(torch.sigmoid(xuij * config.temperature)).mean()

        loss = main_loss

        if config.dat_loss:
            dat_loss = mse(batch_group_embs, batch_item_dat_embs) + mse(batch_group_dat_embs, batch_pos_embs)
            loss += dat_loss * config.dat_loss_coefficient
        if config.clr_loss != 'none':
            if config.clr_loss == 'mse':
                clr_loss = mse(pos_only_id, pos_only_tag) + mse(neg_only_id, neg_only_tag)
            elif config.clr_loss == 'ce':
                pos_loss = info_nce_loss(pos_only_id, pos_only_tag, pos_label, config.clr_temp)
                neg_loss = info_nce_loss(neg_only_tag, neg_only_tag, neg_label, config.clr_temp)
                clr_loss = (pos_loss + neg_loss) / 2
            loss += clr_loss * config.clr_loss_coefficient

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'training stage: {i + 1}, loss: {loss.item():.6f}')

        if (i + 1) % config.eval_per_iter == 0:
            auc = evaluator.eval_auc(model)
            print(f'training stage: {i + 1}, auc: {auc}')

        if (i + 1) % config.save_per_iter == 0:
            model.save(i + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MovieLens-Rand')
    parser.add_argument('--dat_loss', type=bool, default=True)
    parser.add_argument('--clr_loss', type=str, default='mse')
    args = parser.parse_args()

    config = Config(dataset=args.dataset)
    data_loader = DataLoader(config)
    rating_matrix_train = data_loader.load_rating_matrix(dataset_name='train')
    df_eval_user_test = data_loader.load_eval_data(mode='user', dataset_name='test')
    df_eval_group_test = data_loader.load_eval_data(mode='group', dataset_name='test')
    dataset = Dataset(config=config, rating_matrix=rating_matrix_train)
    evaluator = Evaluator(config=config, rating_matrix_pred=dataset.rating_matrix_pred,
                          rating_matrix=rating_matrix_train, group2members_dict=data_loader.group2members_dict)
    model = Model(config=config, group2members=data_loader.group2members_dict, rating_matrix=rating_matrix_train)
    train(config=config, model=model, dataset=dataset)
