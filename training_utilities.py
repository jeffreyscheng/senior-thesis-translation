import torch.nn
import pandas as pd
import time
from global_variables import *


def train_autoencoder(model, autoencoder_objects, optimizer, criterion, clip, loss_df, num_epochs):
    loss_list = []
    total_num_batches = len(autoencoder_objects['train_data']) * num_epochs / autoencoder_hyperparameters['batch_size']
    while True:
        if model.number_of_batches_seen > total_num_batches:
            break
        for i, batch in enumerate(autoencoder_objects['train_iterator']):
            tick = time.time()
            if model.number_of_batches_seen > total_num_batches:
                break
            src = batch.src
            trg = batch.trg
            optimizer.zero_grad()
            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            loss_list.append({'batch_num': model.number_of_batches_seen, 'loss': float(loss.data)})

            print(time.time() - tick)

            if model.number_of_batches_seen % 2000 == 0:
                # save gru_decoder
                torch.save(model, os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))
                # save losses
            if model.number_of_batches_seen % 100 == 0:
                loss_df = loss_df.append(pd.DataFrame(loss_list), ignore_index=True)
                loss_df.to_csv(os.path.join(fixed_vars['autoencoder_directory'], "loss.csv"))
                loss_list = []
    return model, loss_df


def train_translator(model, translation_objects, optimizer, criterion, clip, loss_df, num_epochs):
    loss_list = []
    total_num_batches = len(translation_objects['train_data']) * num_epochs / autoencoder_hyperparameters['batch_size']
    while True:
        if model.number_of_batches_seen > total_num_batches:
            break
        for i, batch in enumerate(translation_objects['train_iterator']):
            tick = time.time()
            if model.number_of_batches_seen > total_num_batches:
                break
            src = batch.src
            trg = batch.trg
            optimizer.zero_grad()
            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            loss_list.append({'batch_num': model.number_of_batches_seen, 'loss': float(loss.data)})

            print(time.time() - tick)

            if model.number_of_batches_seen % 2000 == 0:
                # save gru_decoder
                torch.save(model, os.path.join(fixed_vars['translator_directory'], "translator.model"))
                # save losses
            if model.number_of_batches_seen % 100 == 0:
                loss_df = loss_df.append(pd.DataFrame(loss_list), ignore_index=True)
                loss_df.to_csv(os.path.join(fixed_vars['translator_directory'], "loss.csv"))
                loss_list = []
    return model, loss_df
