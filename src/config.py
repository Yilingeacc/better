import os

import torch


class Config:
    def __init__(self, dataset):
        # Data
        self.dataset = dataset
        self.data_folder_path = os.path.join('..', 'data', self.dataset)
        self.item_path = os.path.join(self.data_folder_path, 'movies.dat')
        self.user_path = os.path.join(self.data_folder_path, 'users.dat')
        self.group_path = os.path.join(self.data_folder_path, 'groupMember.dat')
        self.saves_folder_path = os.path.join('saves', self.dataset)
        self.dataset_file = 'dataset.txt'

        self.item_num = 7710  # CAMRa2011
        self.user_num = 602  # CAMRa2011
        self.group_num = None
        self.total_group_num = None
        self.tag_num = 18

        # Recommendation system
        self.history_length = 5
        self.top_K_list = [5, 10, 20]

        # neural network parameters
        self.embedding_size = 32
        self.user_tower_input_size = self.user_tower_hidden_size = self.embedding_size * 2
        self.item_tower_input_size = self.item_tower_hidden_size = self.embedding_size * 3
        self.nhead = 2

        # Environment
        self.env_n_components = self.embedding_size
        self.env_tol = 1e-4
        self.env_max_iter = 1000
        self.env_alpha = 0.001
        self.sample_per_iter = 50
        self.num_samples = 10000000
        self.threshold = 0.3

        # Optimization parameters
        self.epoch = 1000000
        self.batch_size = 1024
        self.temperature = 10
        self.learning_rate = 1e-4
        self.weight_decay = 1e-6
        self.dat_loss = True
        self.clr_loss = 'mse'
        self.dat_loss_coefficient = 1.
        self.clr_loss_coefficient = 1.
        self.clr_temp = 0.1

        # Eval
        self.eval_per_iter = 10
        self.save_per_iter = 100
        self.eval_positive_threshold = 0.06

        # GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
