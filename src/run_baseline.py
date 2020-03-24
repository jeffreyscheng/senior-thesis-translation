from global_variables import *
from models import *
import torch.optim as optim
from set_up_translation import get_translation_objects
# from transformers import BertModel
from training_utilities import train_translator
import pandas as pd
import gc
import time

loss_df_list = []
translator_objects = get_translation_objects()

proportions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.1]
for proportion_of_data in proportions:
    print("Began: ", proportion_of_data)

    baseline_model = Transformer(src_vocab_size=len(translator_objects['multilingual_bert_tokenizer'].vocab),
                                 trg_vocab_size=len(translator_objects['english_bert_tokenizer'].vocab)).to(fixed_vars['device'])

    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=baseline_hyperparameters['learning_rate'])
    PAD_IDX = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print("Initialized all torch objects and models.  Now training.")

    model, theta_loss_df = train_translator(baseline_model,
                                            translator_objects,
                                            baseline_optimizer,
                                            criterion,
                                            fixed_vars['gradient_clip'],
                                            baseline_hyperparameters['num_epochs'],
                                            proportion_of_data)
    theta_loss_df['proportion_of_data'] = proportion_of_data
    loss_df_list.append(theta_loss_df)
    del model, baseline_model, baseline_optimizer
    gc.collect()
    # print(time.time() - tick)
loss_df = pd.concat(loss_df_list)
loss_df.to_csv(os.path.join(fixed_vars['baseline_directory'], 'full_baseline_loss.csv'))
