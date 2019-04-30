from models import *
import torch.optim as optim
from set_up_translation import get_translation_objects
import torch.nn
import pandas as pd
import time
from global_variables import *


def train(model, translation_objects, optimizer, criterion, clip, loss_df, num_epochs):
    total_num_batches = len(translation_objects['train_data']) * num_epochs / gru_hyperparameters['batch_size']
    while True:
        if model.decoder.number_of_batches_seen > total_num_batches:
            break;
        for i, batch in enumerate(translation_objects['train_iterator']):
            print("---Batch ", i, "---")
            if model.decoder.number_of_batches_seen > total_num_batches:
                break;
            src = batch.src
            trg = batch.trg

            # print(src.size())

            optimizer.zero_grad()
            output = model(src, trg, 1)

            # print(output.size()) # 23 x 40 x 28996
            # print(src.size()) # 23 x 40

            _, best_guess = torch.max(output, dim=2)
            # print(best_guess.size()) # 23 x 40
            print('Actual: ',
                  translation_objects['english_bert_tokenizer'].convert_ids_to_tokens(src[:, 0].flatten().tolist()))
            print('Predicted: ', translation_objects['english_bert_tokenizer'].convert_ids_to_tokens(
                best_guess[:, 0].flatten().tolist()))

            output = output[1:].view(-1, output.shape[-1])
            # print(output.size())
            trg = trg[1:].view(-1)
            # print(trg.size())
            loss = criterion(output, trg)
            print(float(loss.data))
            # raise ValueError
    return model, loss_df


tick = time.time()
translation_objects = get_translation_objects('.en', '.en')
print("Initialized all training objects.")
if gru_hyperparameters['retrain']:
    gru_decoder = torch.load(os.path.join(fixed_vars['gru_directory'],
                                          "gru_decoder.model"), map_location='cpu')
    loss_df = pd.read_csv(os.path.join(fixed_vars['gru_directory'], "loss.csv"))
else:
    gru_decoder = GRUDecoder(fixed_vars['word_embedding_dim'],
                             translation_objects['english_bert_tokenizer'].vocab,
                             fixed_vars['bert_embedding_dim'],
                             gru_hyperparameters['gru_layers'],
                             gru_hyperparameters['gru_dropout'])
    loss_df = pd.DataFrame(columns=['batch_num', 'loss'])
autoencoder = Autoencoder(translation_objects['bert_encoder'],
                          gru_decoder.to('cpu'),
                          fixed_vars['device']).to('cpu')
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=gru_hyperparameters['learning_rate'])
PAD_IDX = 0
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
print("Initialized all torch objects and models.  Now training.")

train(autoencoder, translation_objects, autoencoder_optimizer, criterion, fixed_vars['gradient_clip'], loss_df,
      gru_hyperparameters['num_epochs'])
print(time.time() - tick)
