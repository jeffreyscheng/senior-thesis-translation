from global_variables import *
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator


def get_translation_objects(input_lang, output_lang):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
    bert_encoder.to(fixed_vars['device'])
    bert_encoder.eval()

    def replace_tokens(x):
        if x == '<sos>':
            return 28993
        elif x == '<pad>':
            return 28994
        elif x == '<eos>':
            return 28995
        else:
            return x

    def postprocess_replace_pad(batch, y):
        return [[replace_tokens(token) for token in sentence_list] for sentence_list in batch]

    # originally had <sos> and <eos>
    src_field = Field(tokenize=bert_tokenizer.tokenize,
                      init_token='<sos>',
                      eos_token='<eos>',
                      preprocessing=bert_tokenizer.convert_tokens_to_ids,
                      postprocessing=postprocess_replace_pad,
                      use_vocab=False)  # use_vocab is false because we want Bert.
    trg_field = Field(tokenize=bert_tokenizer.tokenize,
                      init_token='<sos>',
                      eos_token='<eos>')

    train_data, valid_data, test_data = Multi30k.splits(exts=(input_lang, output_lang),
                                                        fields=(src_field, trg_field))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    print(vars(train_data.examples[0]))

    trg_field.build_vocab(train_data, min_freq=2, vectors="glove.6B.100d")
    # print(vars(train_data.examples[0]))

    #     print(f"Unique tokens in source (", input_lang, ") vocabulary: ", len(SRC.vocab))
    # print(f"Unique tokens in target (", output_lang, ") vocabulary: ", len(trg_field.vocab))

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=gru_hyperparameters['batch_size'],
        device=fixed_vars['device'])

    return {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data,
            'train_iterator': train_iterator, 'valid_iterator': valid_iterator, 'test_iterator': test_iterator,
            'bert_tokenizer': bert_tokenizer, 'bert_encoder': bert_encoder,
            'src_field': src_field, 'trg_field': trg_field}
