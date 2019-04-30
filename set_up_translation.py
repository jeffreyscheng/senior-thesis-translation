from global_variables import *
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torchtext.datasets import TranslationDataset, Multi30k
import torchtext.data as data


def get_translation_objects(input_lang, output_lang):
    multilingual_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    english_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_encoder = BertModel.from_pretrained('bert-base-cased')
    bert_encoder.to(fixed_vars['device'])
    bert_encoder.eval()

    def replace_tokens(x):
        if x in ['[PAD]', '<pad>']:
            return 0
        else:
            return x

    def postprocess_replace_pad(batch, y):
        return [[replace_tokens(token) for token in sentence_list] for sentence_list in batch]

    def get_english_field():
        return data.Field(tokenize=english_bert_tokenizer.tokenize,
                          init_token='<pad>',
                          eos_token='<pad>',
                          preprocessing=english_bert_tokenizer.convert_tokens_to_ids,
                          postprocessing=postprocess_replace_pad,
                          use_vocab=False)  # use_vocab is false because we want Bert.

    def get_multilingual_field():
        return data.Field(tokenize=multilingual_bert_tokenizer.tokenize,
                          init_token='<sos>',
                          eos_token='<eos>',
                          preprocessing=multilingual_bert_tokenizer.convert_tokens_to_ids,
                          postprocessing=postprocess_replace_pad,
                          use_vocab=False)  # use_vocab is false because we want Bert.
    if input_lang == '.en':
        src_field = get_english_field()
    else:
        src_field = get_multilingual_field()
    if output_lang == '.en':
        trg_field = get_english_field()
    else:
        trg_field = get_multilingual_field()

    train_data, valid_data, test_data = Multi30k.splits(exts=(input_lang, output_lang),
                                                        fields=(src_field, trg_field))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    # print("Size of target vocab: ", len(test_trg_field.vocab))

    print("Size of target vocab: ", len(english_bert_tokenizer.vocab))

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=gru_hyperparameters['batch_size'],
        device=fixed_vars['device'])

    translation_objects = {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data,
                           'train_iterator': train_iterator, 'valid_iterator': valid_iterator,
                           'test_iterator': test_iterator,
                           'multilingual_bert_tokenizer': multilingual_bert_tokenizer,
                           'english_bert_tokenizer': english_bert_tokenizer,
                           'bert_encoder': bert_encoder, 'src_field': src_field,
                           'trg_field': trg_field}
    return translation_objects
