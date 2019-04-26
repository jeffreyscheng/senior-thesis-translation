import torch.nn
import pandas as pd
import time
from global_variables import *


def train(model, translation_objects, optimizer, criterion, clip, loss_df, num_epochs):
    loss_list = []
    total_num_batches = len(translation_objects['train_data']) * num_epochs / gru_hyperparameters['batch_size']
    while True:
        if model.decoder.number_of_batches_seen > total_num_batches:
            break;
        for i, batch in enumerate(translation_objects['train_iterator']):
            tick = time.time()
            if model.decoder.number_of_batches_seen > total_num_batches:
                break;
            src = batch.src
            featurized_trg = batch
            trg = batch.trg
            #         print("got the right features: ", src.size(), trg.size())

            optimizer.zero_grad()
            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            loss_list.append({'batch_num': model.decoder.number_of_batches_seen, 'loss': float(loss.data)})

            print(time.time() - tick)

            if model.decoder.number_of_batches_seen % 20 == 0:
                # save gru_decoder
                torch.save(model.decoder, os.path.join(fixed_vars['root_directory'], "gru-" + fixed_vars['model_number'], "gru_decoder.model"))
                # save losses
                loss_df = loss_df.append(pd.DataFrame(loss_list), ignore_index=True)
                loss_df.to_csv(os.path.join(fixed_vars['root_directory'], "gru-" + fixed_vars['model_number'], "loss.csv"))
                loss_list = []
    return model, loss_df
