from models import Autoencoder
from set_up_translation import *
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
autoencoder = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"),
                         map_location=device)

autoencoder_objects = get_autoencoder_objects()
translation_objects = get_translation_objects()
translator = torch.load(os.path.join(fixed_vars['translator_directory'], "translator.model"),
                         map_location=device)

# example_sentence = "Hey Chris, here's an example sentence that I'm hoping to recreate."
# example_sentence = "This is an image of a bird, which is an animal that I quite like."
# example_sentence = "Horrible"

example_sentence = "A brown dog is drinking water out of a bowl."
german_example = "Ein brauner Hund trinkt Wasser aus einer Schüssel."
# example_sentence = "Hey Chris, I think that this work is pretty good."


def test_reconstruction(utterance, model, translator=True, trg_utterance=None):
    if not translator:
        e_tokenizer = autoencoder_objects['english_bert_tokenizer']
        src_field = autoencoder_objects['src_field']
        src = autoencoder_objects['trg_field'].process([src_field.preprocess(e_tokenizer.tokenize(utterance))]).to(device)
        output = model(src, src, 1).to(device)
    if translator:
        g_tokenizer = translation_objects['german_bert_tokenizer']
        e_tokenizer = translation_objects['english_bert_tokenizer']
        src_field = translation_objects['src_field']
        trg_field = translation_objects['trg_field']
        src = src_field.process([src_field.preprocess(g_tokenizer.tokenize(utterance))]).to(device)
        trg = trg_field.process([trg_field.preprocess(e_tokenizer.tokenize(trg_utterance))]).to(device)
        output = model(src, trg, 1).to(device)  # figure out trg
    _, best_guess = torch.max(output, dim=2)
    print(best_guess)

    ## testing for Hello, world!
    # print(output.size())
    # print(output[0, 0, 0])
    # print(output[1, 0, 19082])
    # print(output[2, 0, 117])
    # print(output[3, 0, 1362])
    # print(output[4, 0, 106])
    # print(output[4, 0, 0])

    print('Predicted: ', e_tokenizer.convert_ids_to_tokens(best_guess.permute(1, 0).flatten().tolist()))

    print('Actual: ', e_tokenizer.convert_ids_to_tokens(src.flatten().tolist()))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    eval_output = output[1:].view(-1, output.shape[-1])
    trg = src[1:].view(-1)
    print(eval_output)
    print(trg)
    print(criterion(eval_output, trg))


test_reconstruction(german_example, model=translator, translator=True, trg_utterance=example_sentence)
