from global_variables import *
from models import Autoencoder
from set_up_translation import get_translation_objects

gru_decoder = torch.load(os.path.join(fixed_vars['root_directory'], "gru-5", "gru_decoder.model"), map_location='cpu')

translation_objects = get_translation_objects('.en', '.en')
# example_sentence = "Hey Chris, here's an example sentence that I'm hoping to recreate."
example_sentence = "Hello, world!"


# example_sentence = "Horrible"


def test_reconstruction(decoder, utterance):
    autoencoder = Autoencoder(translation_objects['bert_encoder'],
                              decoder,
                              fixed_vars['device']).to(fixed_vars['device'])
    e = translation_objects['english_bert_tokenizer']
    s = translation_objects['src_field']
    src = translation_objects['trg_field'].process([s.preprocess(e.tokenize(utterance))]).to(fixed_vars['device'])
    output = autoencoder(src, src, 1)
    _, best_guess = torch.max(output, dim=2)
    print('Predicted: ', e.convert_ids_to_tokens(best_guess.permute(1, 0)))
    print('Actual: ', src)


test_reconstruction(gru_decoder, example_sentence)
