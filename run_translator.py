from global_variables import *
from models import *
import torch.optim as optim
from set_up_translation import get_translation_objects
from transformers import BertModel
from training_utilities import train_translator
import pandas as pd
import time

loss_df_list = []
translator_objects = get_translation_objects()

for theta in range(21):
    proportion_of_data = theta / 20
    print("Began: ", proportion_of_data)

    # tick = time.time()
    # print("Initialized all training objects.")
    # if translator_hyperparameters['retrain']:
    #     translator = torch.load(os.path.join(fixed_vars['translator_directory'], "translator.model"))
    #     loss_df = pd.read_csv(os.path.join(fixed_vars['translator_directory'], "loss.csv"))
    # else:
    bert_encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
    bert_encoder.to(fixed_vars['device'])
    bert_encoder.train()
    autoencoder = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))

    print("created new encoder + decoder")
    translator = Translator(bert_encoder, autoencoder.decoder, fixed_vars['device']).to(fixed_vars['device'])
    translator_optimizer = optim.Adam(translator.parameters(), lr=translator_hyperparameters['learning_rate'])
    PAD_IDX = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print("Initialized all torch objects and models.  Now training.")

    model, theta_loss_df = train_translator(translator,
                                            translator_objects,
                                            translator_optimizer,
                                            criterion,
                                            fixed_vars['gradient_clip'],
                                            translator_hyperparameters['num_epochs'],
                                            proportion_of_data)
    theta_loss_df['proportion_of_data'] = proportion_of_data
    loss_df_list.append(theta_loss_df)
    # print(time.time() - tick)
loss_df = pd.concat(loss_df_list)
loss_df.to_csv(os.path.join(fixed_vars['translator_directory'], 'full_translator_loss.csv'))
