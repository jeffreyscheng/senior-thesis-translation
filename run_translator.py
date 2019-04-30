from global_variables import *
from models import *
import torch.optim as optim
from set_up_translation import get_translation_objects
from training_utilities import train_translator
import pandas as pd
import time

tick = time.time()
translator_objects = get_translation_objects()
print("Initialized all training objects.")
if translator_hyperparameters['retrain']:
    translator = torch.load(os.path.join(fixed_vars['translator_directory'], "translator.model"))
    loss_df = pd.read_csv(os.path.join(fixed_vars['translator_directory'], "loss.csv"))
else:
    autoencoder = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))
    loss_df = pd.DataFrame(columns=['batch_num', 'loss'])
    print("created new encoder + decoder")
    translator = Translator(autoencoder, fixed_vars['device']).to(fixed_vars['device'])
translator_optimizer = optim.Adam(translator.parameters(), lr=translator_hyperparameters['learning_rate'])
PAD_IDX = 0
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
print("Initialized all torch objects and models.  Now training.")

train_translator(translator,
                 translator_objects,
                 translator_optimizer,
                 criterion,
                 fixed_vars['gradient_clip'],
                 loss_df,
                 translator_hyperparameters['num_epochs'])
print(time.time() - tick)
