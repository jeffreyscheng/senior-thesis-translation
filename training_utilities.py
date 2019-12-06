import torch.nn
from torch.utils.data import Dataset
import torchtext.data as data
import numpy as np
from bleu import *
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
    rows = []
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
            # print('Time: ', time.time() - tick, ', Training Loss: ', loss.data)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        losses = []
        with torch.no_grad():
            for i, batch in enumerate(autoencoder_objects['valid_iterator']):
                src = batch.src
                trg = batch.trg
                optimizer.zero_grad()
                output = model(src, trg)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                losses.append(loss.data)

            validation_loss = float(sum(losses) / len(losses))
            rows.append({'batch_num': model.number_of_batches_seen, 'validation_loss': validation_loss})

            print('Time: ', time.time() - tick, ', Validation loss: ', validation_loss)

            torch.save(model, os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))
    loss_df = loss_df.append(pd.DataFrame(rows), ignore_index=True)
    loss_df.to_csv(fixed_vars['autoencoder_directory'] + "/loss.csv")
    return model, loss_df


def train_translator(model, translation_objects, optimizer, criterion, clip, num_epochs, theta=1):
    loss_df = pd.DataFrame(columns=['batch_num', 'validation_loss', 'bleu'])
    loss_list = []
    t = translation_objects['train_iterator']
    num_batches_per_epoch = int(theta * len(t))

    train_set = list(enumerate(t))[:num_batches_per_epoch]
    for epoch in range(num_epochs):
        for i, batch in train_set:
            tick = time.time()
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

        losses = []
        with torch.no_grad():
            for i, batch in enumerate(translation_objects['valid_iterator']):
                src = batch.src
                trg = batch.trg
                optimizer.zero_grad()
                output = model(src, trg)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                losses.append(loss.data)

        tick = time.time()
        stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        with torch.no_grad():
            for i, batch in enumerate(translation_objects['test_iterator']):
                tick = time.time()
                src = batch.src.to(fixed_vars['device'])
                trg = batch.trg.to(fixed_vars['device'])
                output = model(src, trg, 1)
                output = output[1:]
                _, best_guess = torch.max(output, dim=2)
                trg = trg[1:]
                stats += get_bleu(best_guess, trg)
        bleu_score = bleu(stats)
        print('Epoch: ', epoch, 'Time: ', time.time() - tick, "bleu:", str(bleu_score))

        loss_list.append({'batch_num': model.number_of_batches_seen,
                          'validation_loss': float(sum(losses)) / len(losses),
                          'bleu': bleu_score})

        print('Epoch: ', epoch, 'Time: ', time.time() - tick, ', loss: ', loss.data)
        torch.save(model, os.path.join(fixed_vars['translator_directory'], str(theta) + "translator.model"))
        loss_df = loss_df.append(pd.DataFrame(loss_list), ignore_index=True)
        loss_df.to_csv(os.path.join(fixed_vars['translator_directory'], str(theta) + "loss.csv"))
        loss_list = []
    return model, loss_df
