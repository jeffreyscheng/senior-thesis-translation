from global_variables import *
from models import Autoencoder
from set_up_translation import get_translation_objects
import torch.nn as nn

gru_decoder = torch.load(os.path.join(fixed_vars['root_directory'], "gru-8", "gru_decoder.model"), map_location='cpu')

translation_objects = get_translation_objects('.en', '.en')
# example_sentence = "Hey Chris, here's an example sentence that I'm hoping to recreate."
# example_sentence = "This is an image of a bird, which is an animal that I quite like."
# example_sentence = "Horrible"

example_sentence = "A brown dog is drinking water out of a bowl."


def test_reconstruction(decoder, utterance):
    autoencoder = Autoencoder(translation_objects['bert_encoder'],
                              decoder,
                              fixed_vars['device']).to(fixed_vars['device'])
    e = translation_objects['english_bert_tokenizer']
    s = translation_objects['src_field']
    src = translation_objects['trg_field'].process([s.preprocess(e.tokenize(utterance))]).to(fixed_vars['device'])
    output = autoencoder(src, src, 1)
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


test_reconstruction(gru_decoder, example_sentence)
