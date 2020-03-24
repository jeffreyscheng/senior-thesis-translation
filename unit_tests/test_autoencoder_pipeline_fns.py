import unittest
from src.pipelines import EnglishAutoencoder
from src.set_up_translation import english_field, english_bert_tokenizer
import torch
import os
from src.global_variables import fixed_vars


autoencoder_model = torch.load(os.path.join(fixed_vars['autoencoder_directory'], "autoencoder.model"))
ae = EnglishAutoencoder(english_bert_tokenizer, english_field, autoencoder_model)
example_sentence = 'The bird is drinking water from the fountain.'
example_tokens = english_bert_tokenizer.tokenize(example_sentence)
example_token_ids = english_bert_tokenizer.convert_tokens_to_ids(example_tokens)


class MyTestCase(unittest.TestCase):
    def test_something(self):
        output_token_ids = ae.token_ids_to_token_ids(example_token_ids)
        print(example_token_ids)
        print(output_token_ids)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
