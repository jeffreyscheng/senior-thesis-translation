import torch
import os


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


fixed_vars = {'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'root_directory': os.path.dirname(__file__),
              'bert_embedding_dim': 768,
              'word_embedding_dim': 100,
              'gradient_clip': 1
              }

# attempt 0
# gru_hyperparameters = {'batch_size': 20,
#                        'gru_layers': 1,
#                        'gru_dropout': 0.2,
#                        'learning_rate': 0.001,
#                        'retrain': True,
#                        'num_epochs': 200}

# attempt 1
gru_hyperparameters = {'batch_size': 40,
                       'gru_layers': 1,
                       'gru_dropout': 0.5,
                       'learning_rate': 0.0001,
                       'retrain': True,
                       'num_epochs': 200}
