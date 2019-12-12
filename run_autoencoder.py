from global_variables import *
from models import *
import torch.optim as optim
from set_up_translation import get_autoencoder_objects
from transformers import BertModel, RobertaModel
from training_utilities import train_autoencoder
import pandas as pd
import time

tick = time.time()
autoencoder_objects = get_autoencoder_objects()
print("Initialized all training objects.")
if autoencoder_hyperparameters['retrain']:
    autoencoder = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))
    loss_df = pd.read_csv(os.path.join(fixed_vars['autoencoder_directory'], "loss.csv"))
else:
    if autoencoder_hyperparameters['roberta']:
        bert_encoder = RobertaModel.from_pretrained('roberta-base')
    else:
        bert_encoder = BertModel.from_pretrained('bert-base-cased')
    bert_encoder.to(fixed_vars['device'])
    bert_encoder.eval()
    gru_decoder = GRUDecoder(fixed_vars['word_embedding_dim'],
                             autoencoder_objects['english_bert_tokenizer'].vocab,
                             fixed_vars['bert_embedding_dim'],
                             autoencoder_hyperparameters['gru_layers'],
                             autoencoder_hyperparameters['gru_dropout'])
    loss_df = pd.DataFrame(columns=['batch_num', 'validation_loss', 'bleu'])
    print("created new encoder + decoder")
    autoencoder = Autoencoder(bert_encoder,
                              gru_decoder,
                              fixed_vars['device']).to(fixed_vars['device'])
autoencoder_optimizer = optim.Adam(autoencoder.decoder.parameters(),
                                   lr=autoencoder_hyperparameters['learning_rate'],
                                   weight_decay=10 ** (-5))
PAD_IDX = 0
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
print("Initialized all torch objects and models.  Now training.")

train_autoencoder(autoencoder,
                  autoencoder_objects,
                  autoencoder_optimizer,
                  criterion,
                  fixed_vars['gradient_clip'],
                  loss_df,
                  autoencoder_hyperparameters['num_epochs'])
print(time.time() - tick)
