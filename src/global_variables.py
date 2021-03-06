import torch
import os


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


fixed_vars = {'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'root_directory': os.path.split(os.path.dirname(__file__))[0],
              'bert_embedding_dim': 768,
              'word_embedding_dim': 100,
              'gradient_clip': 10,
              'autoencoder_model_number': "0",
              'translator_model_number': "0",
              'baseline_model_number': "0"}

fixed_vars['autoencoder_directory'] = os.path.join(fixed_vars['root_directory'], 'results',
                                                   'autoencoder-' + fixed_vars['autoencoder_model_number'])
fixed_vars['translator_directory'] = os.path.join(fixed_vars['root_directory'], 'results',
                                                  'translator-' + fixed_vars['translator_model_number'])
fixed_vars['baseline_directory'] = os.path.join(fixed_vars['root_directory'], 'results',
                                                'baseline-' + fixed_vars['baseline_model_number'])

safe_mkdir(fixed_vars['autoencoder_directory'])
safe_mkdir(fixed_vars['translator_directory'])
safe_mkdir(fixed_vars['baseline_directory'])

# attempt 0: 100 word embedding, 0.8 dropout
# attempt 1: 200 word embedding, 0.8 dropout
# attempt 2: 100 word embedding, 0.2 dropout
autoencoder_hyperparameters = {'batch_size': 25,
                               'gru_layers': 2,
                               'gru_dropout': 0.2,
                               'learning_rate': 0.0001,
                               'restart': False,
                               'num_epochs': 500,
                               'roberta': True}

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
                              'restart': False,
                              'num_epochs': 400}

baseline_hyperparameters = {'batch_size': 40,
                            'learning_rate': 0.0001,
                            'num_epochs': 200}
