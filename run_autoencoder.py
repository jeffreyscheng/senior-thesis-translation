from global_variables import *
from models import *
import torch.optim as optim
from set_up_translation import get_translation_objects
from training_utilities import train
import pandas as pd
import time

tick = time.time()
translation_objects = get_translation_objects('.en', '.en')
print("Initialized all training objects.")
if gru_hyperparameters['retrain']:
    gru_decoder = torch.load(os.path.join(fixed_vars['root_directory'],
                                          "gru-" + str(fixed_vars['model_number']),
                                          "gru_decoder.model"))
    loss_df = pd.read_csv(os.path.join(fixed_vars['root_directory'], "gru-" + str(fixed_vars['model_number']), "loss.csv"))
else:
    gru_decoder = GRUDecoder(fixed_vars['word_embedding_dim'],
                             translation_objects['english_bert_tokenizer'].vocab,
                             fixed_vars['bert_embedding_dim'],
                             gru_hyperparameters['gru_layers'],
                             gru_hyperparameters['gru_dropout'])
    loss_df = pd.DataFrame(columns=['batch_num', 'loss'])
autoencoder = Autoencoder(translation_objects['bert_encoder'],
                          gru_decoder,
                          fixed_vars['device']).to(fixed_vars['device'])
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=gru_hyperparameters['learning_rate'])
<<<<<<< HEAD
PAD_IDX = 28994
=======
PAD_IDX = translation_objects['trg_field'].vocab.stoi['<pad>']
>>>>>>> 1e61fb0908658fcaf7ea0c9bd4bcb8d2b8a6bffe
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
print("Initialized all torch objects and models.  Now training.")

train(autoencoder, translation_objects, autoencoder_optimizer, criterion, fixed_vars['gradient_clip'], loss_df, gru_hyperparameters['num_epochs'])
print(time.time() - tick)
