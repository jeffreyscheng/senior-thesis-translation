from src.global_variables import *
from transformers import BertTokenizer, RobertaTokenizer
from torchtext.datasets import Multi30k
import torchtext.data as data


german_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
if autoencoder_hyperparameters['roberta']:
    english_bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
else:
    english_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

ger_pad_index = german_bert_tokenizer.convert_tokens_to_ids(german_bert_tokenizer.pad_token)
ger_eos_index = german_bert_tokenizer.convert_tokens_to_ids(german_bert_tokenizer.eos_token)
ger_cls_index = german_bert_tokenizer.convert_tokens_to_ids(german_bert_tokenizer.cls_token)
ger_unk_index = german_bert_tokenizer.convert_tokens_to_ids(german_bert_tokenizer.unk_token)
pad_index = english_bert_tokenizer.convert_tokens_to_ids(english_bert_tokenizer.pad_token)
eos_index = english_bert_tokenizer.convert_tokens_to_ids(english_bert_tokenizer.eos_token)
cls_index = english_bert_tokenizer.convert_tokens_to_ids(english_bert_tokenizer.cls_token)
unk_index = english_bert_tokenizer.convert_tokens_to_ids(english_bert_tokenizer.unk_token)

english_field = data.Field(tokenize=english_bert_tokenizer.tokenize,
                           init_token=cls_index,
                           eos_token=eos_index,
                           pad_token=pad_index,
                           unk_token=unk_index,
                           preprocessing=english_bert_tokenizer.convert_tokens_to_ids,
                           # postprocessing=postprocess_replace_pad,  #  I think this is useless?
                           use_vocab=False)  # use_vocab is false because we want Bert.

train_data, valid_data, test_data = Multi30k.splits(exts=('.en', '.en'),
                                                    fields=(english_field, english_field))

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=autoencoder_hyperparameters['batch_size'],
    device=fixed_vars['device'])

autoencoder_objects = {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data,
                       'train_iterator': train_iterator, 'valid_iterator': valid_iterator,
                       'test_iterator': test_iterator,
                       'english_bert_tokenizer': english_bert_tokenizer,
                       'english_field': english_field}

german_field = data.Field(tokenize=german_bert_tokenizer.tokenize,
                          init_token=ger_cls_index,
                          eos_token=ger_eos_index,
                          pad_token=ger_pad_index,
                          unk_token=ger_unk_index,
                          preprocessing=german_bert_tokenizer.convert_tokens_to_ids,
                          use_vocab=False)  # use_vocab is false because we want Bert.

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german_field, english_field))

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=autoencoder_hyperparameters['batch_size'],
    device=fixed_vars['device'])

translation_objects = {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data,
                       'train_iterator': train_iterator, 'valid_iterator': valid_iterator,
                       'test_iterator': test_iterator,
                       'english_bert_tokenizer': english_bert_tokenizer,
                       'german_bert_tokenizer': german_bert_tokenizer,
                       'english_field': english_field, 'german_field': german_field}
