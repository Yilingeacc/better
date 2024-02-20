import os
import numpy as np
from sklearn.decomposition import NMF
from config import Config
from data import DataLoader


class Dataset:
    def __init__(self, config, rating_matrix):
        self.rating_matrix_pred = None
        self.calc_rating_matrix_pred(config, rating_matrix)
        self.data = self.generate_pairwise_training_set(config=config)
        self.pairs = self.data.shape[0]

    def calc_rating_matrix_pred(self, config, rating_matrix):
        env_name = 'env_' + config.dataset + '_' + str(config.env_n_components) + '.npy'
        env_path = os.path.join(config.saves_folder_path, env_name)

        if not os.path.exists(env_path):
            env_model = NMF(n_components=config.env_n_components, init='random', tol=config.env_tol,
                            max_iter=config.env_max_iter, verbose=True,
                            random_state=0)
            print('-' * 50)
            print('Train environment:')
            W = env_model.fit_transform(X=rating_matrix)
            H = env_model.components_
            self.rating_matrix_pred = W @ H
            print('-' * 50)
            np.save(env_path, self.rating_matrix_pred)
            print(f'Save environment: {env_path}')
        else:
            self.rating_matrix_pred = np.load(env_path)
            print(f'Load environment: {env_path}')

    def generate_pairwise_training_set(self, config):
        pairwise_training_set = []
        if os.path.exists(config.dataset_file):
            print(f'Load dataset: {config.dataset_file}')
            with open(config.dataset_file, 'r') as f:
                line = f.readline()[:-1]
                while line:
                    pairwise_training_set.append([int(i) for i in line.split()])
                    line = f.readline()[:-1]
        else:
            num_users = config.total_group_num
            num_items = config.item_num

            available_user = [i for i in range(1, num_users + 1) if any(self.rating_matrix_pred[i] > 0)]
            available_item = [i for i in range(1, num_items + 1) if any(self.rating_matrix_pred[:, i] > 0)]
            p_dict = {}
            for user_id in available_user:
                p_dict[user_id] = self.rating_matrix_pred[user_id] / np.sum(self.rating_matrix_pred[user_id])

            while len(pairwise_training_set) < config.num_samples:
                user_id = np.random.choice(available_user)

                positive_item = np.random.choice(num_items + 1, size=config.sample_per_iter, p=p_dict[user_id])
                negative_item = np.random.choice(available_item, size=config.sample_per_iter)

                for pos, neg in zip(positive_item, negative_item):
                    if self.rating_matrix_pred[user_id, pos] < self.rating_matrix_pred[user_id, neg]:
                        pos, neg = neg, pos
                    positive_score = self.rating_matrix_pred[user_id, pos]
                    negative_score = self.rating_matrix_pred[user_id, neg]

                    if positive_score - negative_score > config.threshold:
                        pairwise_training_set.append([user_id, pos, neg])

        return np.array(pairwise_training_set)

    def sample(self, batch_size):
        indices = np.random.choice(self.pairs, size=batch_size, replace=False)
        return self.data[indices]


if __name__ == '__main__':
    config = Config(dataset='MovieLens-Rand')
    data_loader = DataLoader(config)
    rating_matrix_train = data_loader.load_rating_matrix(dataset_name='train')
    dataset = Dataset(config=config, rating_matrix=rating_matrix_train)


