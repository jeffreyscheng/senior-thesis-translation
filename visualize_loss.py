import pandas as pd
from global_variables import *
import matplotlib.pyplot as plt
from models import Autoencoder
from set_up_translation import get_translation_objects

loss_df = pd.read_csv(os.path.join(fixed_vars['root_directory'], "gru-" + fixed_vars['model_number'], "loss.csv"))
loss_df['rolling_loss'] = loss_df['loss'].rolling(window=1000).mean()
# gca stands for 'get current axis'
ax = plt.gca()
loss_df.loc[200000:, ].plot(kind='line', x='batch_num', y='rolling_loss', ax=ax)
plt.show()

gru_decoder = torch.load(os.path.join(fixed_vars['root_directory'],
                                      "gru-" + str(fixed_vars['model_number']),
                                      "gru_decoder.model"), map_location='cpu')

translation_objects = get_translation_objects('.en', '.en')
# example_sentence = "Hey Chris, here's an example sentence that I'm hoping to recreate."
example_sentence = "Hello, world!"
# example_sentence = "Horrible"


def test_reconstruction(decoder, utterance):
    autoencoder = Autoencoder(translation_objects['bert_encoder'],
                              decoder,
                              fixed_vars['device']).to(fixed_vars['device'])
    b = translation_objects['bert_tokenizer']
    s = translation_objects['src_field']
    t = translation_objects['trg_field']
    src = translation_objects['src_field'].process([s.preprocess(b.tokenize(utterance))])
    trg = translation_objects['trg_field'].process([t.preprocess(utterance)])
    output = autoencoder(src, trg, 1)
    _, best_guess = torch.max(output, dim=2)
    print('Predicted: ', translation_objects['trg_field'].reverse(best_guess.permute(1, 0)))
    print('Actual: ', translation_objects['trg_field'].reverse(trg))


test_reconstruction(gru_decoder, example_sentence)
