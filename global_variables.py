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
              'autoencoder_model_number': "1",
              'translator_model_number': "0",
              'baseline_model_number': "0"}

fixed_vars['autoencoder_directory'] = os.path.join(fixed_vars['root_directory'],
                                                   'autoencoder-' + fixed_vars['autoencoder_model_number'])
fixed_vars['translator_directory'] = os.path.join(fixed_vars['root_directory'],
                                                  'translator-' + fixed_vars['translator_model_number'])
fixed_vars['baseline_directory'] = os.path.join(fixed_vars['root_directory'],
                                                'baseline-' + fixed_vars['baseline_model_number'])

safe_mkdir(fixed_vars['autoencoder_directory'])
safe_mkdir(fixed_vars['translator_directory'])
safe_mkdir(fixed_vars['baseline_directory'])

# attempt 0
autoencoder_hyperparameters = {'batch_size': 40,
                               'gru_layers': 1,
                               'gru_dropout': 0.8,
                               'learning_rate': 0.0001,
                               'retrain': True,
                               'num_epochs': 300}

# attempt 0
# translator_hyperparameters = {'batch_size': 40,
#                               'translator_dropout': 0.8,
#                               'learning_rate': 0.0001,
#                               'retrain': False,
#                               'num_epochs': 200}

# attempt 1
translator_hyperparameters = {'batch_size': 40,
                              'translator_dropout': 0.8,
                              'learning_rate': 0.0001,
                              'retrain': True,
                              'num_epochs': 400}


baseline_hyperparameters = {'batch_size': 40,
                            'learning_rate': 0.0001,
                            'num_epochs': 200}
