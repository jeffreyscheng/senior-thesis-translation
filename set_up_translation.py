from global_variables import *
from transformers import BertTokenizer, RobertaTokenizer
from torchtext.datasets import TranslationDataset, Multi30k
import torchtext.data as data


def replace_tokens(x):
    if x in ['[PAD]', '<pad>']:
        return 0
    else:
        return x


def postprocess_replace_pad(batch, y):
    def truncate_sentence(sent):
        if len(sent) > 100:
            return sent[:100]
        return sent
    return [[replace_tokens(token) for token in truncate_sentence(sentence_list)] for sentence_list in batch]


def get_autoencoder_objects():
    if autoencoder_hyperparameters['roberta']:
        english_bert_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        english_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def get_english_field():
        return data.Field(tokenize=english_bert_tokenizer.tokenize,
                          init_token=english_bert_tokenizer._cls_token,
                          eos_token=english_bert_tokenizer._eos_token,
                          preprocessing=english_bert_tokenizer.convert_tokens_to_ids,
                          postprocessing=postprocess_replace_pad,
                          use_vocab=False)  # use_vocab is false because we want Bert.

    src_field = get_english_field()
    trg_field = get_english_field()

    train_data, valid_data, test_data = Multi30k.splits(exts=('.en', '.en'),
                                                        fields=(src_field, trg_field))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    print("Size of target vocab: ", len(english_bert_tokenizer.vocab))

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=autoencoder_hyperparameters['batch_size'],
        device=fixed_vars['device'])

    autoencoder_objects = {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data,
                           'train_iterator': train_iterator, 'valid_iterator': valid_iterator,
                           'test_iterator': test_iterator,
                           'english_bert_tokenizer': english_bert_tokenizer,
                           'src_field': src_field, 'trg_field': trg_field}
    return autoencoder_objects


def get_translation_objects():
    german_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    english_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    def get_german_field():
        return data.Field(tokenize=german_bert_tokenizer.tokenize,
                          init_token=german_bert_tokenizer.cls_token,
                          eos_token=german_bert_tokenizer.eos_token,
                          preprocessing=german_bert_tokenizer.convert_tokens_to_ids,
                          postprocessing=postprocess_replace_pad,
                          use_vocab=False)  # use_vocab is false because we want Bert.

    def get_english_field():
        return data.Field(tokenize=english_bert_tokenizer.tokenize,
                          init_token=english_bert_tokenizer._cls_token,
                          eos_token=english_bert_tokenizer._eos_token,
                          preprocessing=english_bert_tokenizer.convert_tokens_to_ids,
                          postprocessing=postprocess_replace_pad,
                          use_vocab=False)  # use_vocab is false because we want Bert.

    src_field = get_german_field()
    trg_field = get_english_field()

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(src_field, trg_field))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    print("Size of target vocab: ", len(english_bert_tokenizer.vocab))

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=autoencoder_hyperparameters['batch_size'],
        device=fixed_vars['device'])

    translation_objects = {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data,
                           'train_iterator': train_iterator, 'valid_iterator': valid_iterator,
                           'test_iterator': test_iterator,
                           'english_bert_tokenizer': english_bert_tokenizer,
                           'german_bert_tokenizer': german_bert_tokenizer,
                           'src_field': src_field, 'trg_field': trg_field}
    return translation_objects

