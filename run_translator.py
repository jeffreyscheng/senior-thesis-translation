from global_variables import *
from models import *
import torch.optim as optim
from set_up_translation import get_translation_objects
from transformers import BertModel
from training_utilities import train_translator
import pandas as pd
import gc
import time

loss_df_list = []
translator_objects = get_translation_objects()

proportions = [1]
# proportions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
for proportion_of_data in proportions:
    print("Began: ", proportion_of_data)

    # tick = time.time()
    # print("Initialized all training objects.")
    # if translator_hyperparameters['retrain']:
    #     translator = torch.load(os.path.join(fixed_vars['translator_directory'], "translator.model"))
    #     loss_df = pd.read_csv(os.path.join(fixed_vars['translator_directory'], "loss.csv"))
    # else:
    bert_encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
    bert_encoder.to(fixed_vars['device'])
    # bert_encoder.train()
    autoencoder = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))

    print("created new encoder + decoder")
    translator = Translator(bert_encoder, autoencoder.decoder, fixed_vars['device']).to(fixed_vars['device'])
    translator_optimizer = optim.Adam(list(translator.decoder.parameters()) + list(translator.fc1.parameters()) + list(translator.fc2.parameters()),
                                      lr=translator_hyperparameters['learning_rate'],
                                      weight_decay=10 ** (-5))
    PAD_IDX = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print("Initialized all torch objects and models.  Now training.")

    translator.encoder.eval()
    translator.decoder.train()
    translator.fc1.train()
    translator.fc2.train()
    model, theta_loss_df = train_translator(translator,
                                            translator_objects,
                                            translator_optimizer,
                                            criterion,
                                            fixed_vars['gradient_clip'],
                                            translator_hyperparameters['num_epochs'],
                                            proportion_of_data)
    theta_loss_df['proportion_of_data'] = proportion_of_data
    loss_df_list.append(theta_loss_df)
    del model, translator, translator_optimizer
    gc.collect()
    # print(time.time() - tick)
loss_df = pd.concat(loss_df_list)
loss_df.to_csv(os.path.join(fixed_vars['translator_directory'], 'full_translator_loss.csv'))
