import torch
import os


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


fixed_vars = {'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'root_directory': os.path.dirname(__file__),
              'bert_embedding_dim': 768,
              'word_embedding_dim': 100,
              'gradient_clip': 1,
              'gru_model_number': "4",
              'ffn_model_number': "0"}

fixed_vars['gru_directory'] = os.path.join(fixed_vars['root_directory'], 'gru-' + fixed_vars['gru_model_number'])
fixed_vars['ffn_directory'] = os.path.join(fixed_vars['root_directory'], 'ffn-' + fixed_vars['ffn_model_number'])

safe_mkdir(fixed_vars['gru_directory'])
safe_mkdir(fixed_vars['ffn_directory'])


# attempt 0
# gru_hyperparameters = {'batch_size': 20,
#                        'gru_layers': 1,
#                        'gru_dropout': 0.2,
#                        'learning_rate': 0.001,
#                        'retrain': True,
#                        'num_epochs': 200}

# attempt 1
# 40, 1, 0.5, 0.0001, False, 500

# attempt 2
# 40, 1, 0.5, 0.0003, False, 500

# attempt 3
# same but with right tokenizer :(

# attempt 4
gru_hyperparameters = {'batch_size': 40,
                       'gru_layers': 1,
                       'gru_dropout': 0.5,
                       'learning_rate': 0.0001,
                       'retrain': False,
                       'num_epochs': 500}
