import argparse
import torch
import numpy as np

from config import Config
from data import DataLoader
from dataset import Dataset
from model import Model


class Evaluator:
    def __init__(self, config: Config, rating_matrix_pred, rating_matrix, group2members_dict):
        self.config = config
        self.threshold = config.eval_positive_threshold
        self.rating_matrix = rating_matrix
        self.rating_matrix_pred = rating_matrix_pred
        self.available_item = [i for i in range(1, config.item_num + 1) if any(rating_matrix_pred[:, i] > 0)]
        self.group2members_dict = group2members_dict

    @staticmethod
    def calc_auc(label, predict):
        combined = [(l, p) for l, p in zip(label, predict)]
        combined.sort(key=lambda x: x[1], reverse=True)
        auc, pos, neg = 0, 0, 0

        for l, _ in combined:
            if l:
                pos += 1
            else:
                auc += pos
                neg += 1
        return auc / (pos * neg)

    def eval_auc(self, model: Model):
        with torch.no_grad():
            user_aucs = []
            group_aucs = []
            user_tensors = model.users(list(range(1, self.config.total_group_num + 1)))[0]
            item_tensors = model.items(list(range(self.config.item_num + 1)))[0]
            for i, _ in enumerate(range(1, self.config.total_group_num)):
                items = self.rating_matrix[i].indices
                if self.rating_matrix[i, items].data.sum() in [0, len(self.rating_matrix[i, items].data)]:
                    continue
                user_tensor = user_tensors[i]
                item_tensor = item_tensors[items]
                scores = torch.matmul(user_tensor, item_tensor.T)
                auc = self.calc_auc(self.rating_matrix[i, items].data, scores)
                if len(self.group2members_dict[i]) == 1:
                    user_aucs.append(auc)
                else:
                    group_aucs.append(auc)
        return sum(user_aucs) / len(user_aucs), sum(group_aucs) / len(group_aucs)

    def eval_hr_ndcg(self, model: Model, data):
        num_sample = len(data)
        with torch.no_grad():
            hrs = [0, 0, 0]
            ndcgs = [0, 0, 0]
            user_tensors = model.users(list(range(1, self.config.total_group_num + 1)))[0]
            item_tensors = model.items(list(range(self.config.item_num + 1)))[0]
            for sample in data.values:
                user_tensor = user_tensors[sample[0]]
                item_tensor = item_tensors[[sample[2], *sample[3]]]
                scores = torch.matmul(user_tensor, item_tensor.T)
                rank = torch.where(torch.flip(torch.sort(scores).indices, dims=[0]) == 0)[0].item()
                if rank <= 5:
                    hrs[0] += 1
                    ndcgs[0] += 1 / np.log2(rank + 2)
                if rank <= 10:
                    hrs[1] += 1
                    ndcgs[1] += 1 / np.log2(rank + 2)
                if rank <= 20:
                    hrs[2] += 1
                    ndcgs[2] += 1 / np.log2(rank + 2)
        return [i / num_sample for i in [*hrs, *ndcgs]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MovieLens-Rand')
    parser.add_argument('--test_no', type=int, default=160000)
    args = parser.parse_args()

    config = Config(dataset=args.dataset)
    config.device = torch.device('cpu')
    data_loader = DataLoader(config)
    rating_matrix_train = data_loader.load_rating_matrix(dataset_name='train')
    df_eval_user_test = data_loader.load_eval_data(mode='user', dataset_name='test')
    df_eval_group_test = data_loader.load_eval_data(mode='group', dataset_name='test')
    dataset = Dataset(config=config, rating_matrix=rating_matrix_train)
    model = Model(config, data_loader.group2members_dict)
    evaluator = Evaluator(config=config, rating_matrix_pred=dataset.rating_matrix_pred,
                          rating_matrix=rating_matrix_train, group2members_dict=data_loader.group2members_dict)

    test_no = 'test'
    # test_no = args.test_no

    model_file = f'./model/MovieLens-Rand/better_{test_no}.pth'
    model.load_state_dict(torch.load(model_file))
    user_auc, group_auc = evaluator.eval_auc(model)
    res_user = evaluator.eval_hr_ndcg(model, df_eval_user_test)
    res_group = evaluator.eval_hr_ndcg(model, df_eval_group_test)
    print(f'{model_file}: user_auc={user_auc}, group_auc={group_auc}')
    print(res_user)
    print(res_group)
