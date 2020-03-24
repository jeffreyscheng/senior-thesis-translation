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

# proportions = [1]
proportions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
for proportion_of_data in proportions:
    print("Began: ", proportion_of_data)

    # tick = time.time()
    # print("Initialized all training objects.")
    # if translator_hyperparameters['retrain']:
    #     translator = torch.load(os.path.join(fixed_vars['translator_directory'], "translator.model"))
    #     loss_df = pd.read_csv(os.path.join(fixed_vars['translator_directory'], "loss.csv"))
    # else:
    # bert_encoder = BertModel.from_pretrained('bert-base-german-cased')
    # bert_encoder.to(fixed_vars['device'])
    # bert_encoder.train()
    autoencoder = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))

    print("created new encoder + decoder")
    translator = Translator(encoder=autoencoder.encoder,
                            decoder=autoencoder.decoder,
                            input_dim=fixed_vars['bert_embedding_dim'],
                            hidden_widths=[1000],
                            output_dim=fixed_vars['bert_embedding_dim'],
                            num_gru_layers=autoencoder_hyperparameters['gru_layers'],
                            device=fixed_vars['device']).to(fixed_vars['device'])
    translator_optimizer = optim.Adam(translator.ffn.parameters(),
                                      lr=translator_hyperparameters['learning_rate'],
                                      weight_decay=10 ** (-5))
    PAD_IDX = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print("Initialized all torch objects and models.  Now training.")

    translator.encoder.eval()
    translator.decoder.train()  # effectively frozen; not in the optimizer
    translator.ffn.train()
    loss_df = pd.DataFrame(columns=['batch_num', 'validation_loss', 'bleu'])
    model, theta_loss_df = train_translator(model=translator,
                                            translation_objects=translator_objects,
                                            optimizer=translator_optimizer,
                                            criterion=criterion,
                                            clip=fixed_vars['gradient_clip'],
                                            original_loss_df=loss_df,
                                            num_epochs=translator_hyperparameters['num_epochs'],
                                            name='translator',
                                            path=fixed_vars['translator_directory'])
                                            # proportion_of_data)
    theta_loss_df['proportion_of_data'] = proportion_of_data
    loss_df_list.append(theta_loss_df)
    del model, translator, translator_optimizer
    gc.collect()
    # print(time.time() - tick)
loss_df = pd.concat(loss_df_list)
loss_df.to_csv(os.path.join(fixed_vars['translator_directory'], 'full_translator_loss.csv'))
