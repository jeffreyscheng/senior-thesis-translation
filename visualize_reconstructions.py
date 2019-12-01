from models import Autoencoder
from set_up_translation import *
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
autoencoder = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"),
                         map_location=device)

autoencoder_objects = get_autoencoder_objects()
# example_sentence = "Hey Chris, here's an example sentence that I'm hoping to recreate."
# example_sentence = "This is an image of a bird, which is an animal that I quite like."
# example_sentence = "Horrible"

example_sentence = "A brown dog is drinking water out of a bowl."
# example_sentence = "Hey Chris, I think that this work is pretty good."

def test_reconstruction(utterance):
    e = autoencoder_objects['english_bert_tokenizer']
    s = autoencoder_objects['src_field']
    src = autoencoder_objects['trg_field'].process([s.preprocess(e.tokenize(utterance))]).to(device)
    output = autoencoder(src, src, 1).to(device)
    _, best_guess = torch.max(output, dim=2)
    print(best_guess)

    ## testing for Hello, world!
    print(output.size())
    print(output[0, 0, 0])
    print(output[1, 0, 19082])
    print(output[2, 0, 117])
    print(output[3, 0, 1362])
    print(output[4, 0, 106])
    print(output[4, 0, 0])

    print('Predicted: ', e.convert_ids_to_tokens(best_guess.permute(1, 0).flatten().tolist()))
    print(src)
    print('Actual: ', e.convert_ids_to_tokens(src.flatten().tolist()))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    eval_output = output[1:].view(-1, output.shape[-1])
    trg = src[1:].view(-1)
    print(eval_output)
    print(trg)
    print(criterion(eval_output, trg))


test_reconstruction(example_sentence)
