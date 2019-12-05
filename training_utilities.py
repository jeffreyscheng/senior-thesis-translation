import torch.nn
from torch.utils.data import Dataset
import torchtext.data as data
import pandas as pd
import time
import random
from global_variables import *


def iter_sample_fast(iterable, sample_size):
    results = []
    iterator = iter(iterable)
    # Fill in the first sample_size elements:
    try:
        for _ in range(sample_size):
            results.append(iterator.__next__())
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, sample_size):
        r = random.randint(0, i)
        if r < sample_size:
            results[r] = v  # at a decreasing rate, replace random items
    return results


def train_autoencoder(model, autoencoder_objects, optimizer, criterion, clip, loss_df, num_epochs):
    loss_list = []
    total_num_batches = len(autoencoder_objects['train_data']) * num_epochs / autoencoder_hyperparameters['batch_size']
    while True:
        if model.number_of_batches_seen > total_num_batches:
            break
        # train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        #     (autoencoder_objects['train_data'],
        #      autoencoder_objects['valid_data'],
        #      autoencoder_objects['test_data']),
        #     batch_size=autoencoder_hyperparameters['batch_size'],
        #     device=fixed_vars['device'])
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

            print('Time: ', time.time() - tick, ', loss: ', loss.data)

            if model.number_of_batches_seen % 2000 == 0:
                # save gru_decoder
                torch.save(model, os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))
                # save losses
            if model.number_of_batches_seen % 100 == 0:
                loss_df = loss_df.append(pd.DataFrame(loss_list), ignore_index=True)
                loss_df.to_csv(fixed_vars['autoencoder_directory'] + "/loss.csv")
                loss_list = []
    return model, loss_df


def train_translator(model, translation_objects, optimizer, criterion, clip, num_epochs, theta=1):
    loss_df = pd.DataFrame(columns=['batch_num', 'loss'])
    loss_list = []
    t = translation_objects['train_iterator']
    num_batches_per_epoch = int(theta * len(t))
    total_num_batches = num_epochs * num_batches_per_epoch

    train_set = list(enumerate(t))[:num_batches_per_epoch]
    while True:
        if model.number_of_batches_seen >= total_num_batches:
            break
        model.train()
        for i, batch in train_set:
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

        model.eval()
        losses = []
        for i, batch in enumerate(translation_objects['valid_iterator']):
            print(batch)
            src = batch.src
            trg = batch.trg
            optimizer.zero_grad()
            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            losses.append(loss.data)

        loss_list.append({'batch_num': model.number_of_batches_seen, 'validation_loss': float(sum(losses)) / len(losses)})

        print('Time: ', time.time() - tick, ', loss: ', loss.data)
        torch.save(model, os.path.join(fixed_vars['translator_directory'], str(theta) + "translator.model"))
        loss_df = loss_df.append(pd.DataFrame(loss_list), ignore_index=True)
        loss_df.to_csv(os.path.join(fixed_vars['translator_directory'], str(theta) + "loss.csv"))
        loss_list = []
    return model, loss_df
