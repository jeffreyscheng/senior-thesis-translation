from global_variables import *
import torch.optim as optim
from set_up_translation import get_autoencoder_objects
from pytorch_pretrained_bert import BertModel
from training_utilities import train_autoencoder
import pandas as pd
from bleu import *
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
autoencoder_objects = get_autoencoder_objects()
autoencoder = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"), map_location=device)

with torch.no_grad():
    for i, batch in enumerate(autoencoder_objects['test_iterator']):
        tick = time.time()
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        output = autoencoder(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        best_guess = torch.max(output, dim=2)
        trg = trg[1:].view(-1)

        print(get_bleu(best_guess, trg))
