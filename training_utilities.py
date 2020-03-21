import torch.nn
from torch.utils.data import Dataset
import torchtext.data as data
import numpy as np
from bleu import *
import pandas as pd
import time
import random
from global_variables import *
import gc



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


def train_translator(model, translation_objects, optimizer, criterion, clip, original_loss_df, num_epochs, path):
    rows = []
    total_num_batches = len(translation_objects['train_data']) * num_epochs / translator_hyperparameters['batch_size']
    loss_df = original_loss_df  # in case you're already done
    while True:
        if model.number_of_batches_seen > total_num_batches:
            break
        tick = time.time()
        for i, batch in enumerate(translation_objects['train_iterator']):
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
            del src, trg, output, loss
        print("Finished training:", time.time() - tick)

        tick = time.time()
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

            validation_loss = float(sum(losses) / len(losses))

            stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
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
            rows.append({'batch_num': model.number_of_batches_seen, 'validation_loss': validation_loss,
                         'bleu': bleu_score})

            print('Finished validating: ', time.time() - tick, ', Validation loss: ', validation_loss, "BlEU:", bleu_score)

            torch.save(model, os.path.join(path, "translator.model"))
        loss_df = original_loss_df.append(pd.DataFrame(rows), ignore_index=True)
        loss_df.to_csv(path + "/loss.csv")
    return model, loss_df