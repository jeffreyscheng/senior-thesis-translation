from global_variables import *
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torchtext.datasets import TranslationDataset, Multi30k
import torchtext.data as data


def get_translation_objects(input_lang, output_lang):
    multilingual_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    english_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
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
    src_field = data.Field(tokenize=multilingual_bert_tokenizer.tokenize,
                           init_token='<sos>',
                           eos_token='<eos>',
                           preprocessing=multilingual_bert_tokenizer.convert_tokens_to_ids,
                           postprocessing=postprocess_replace_pad,
                           use_vocab=False)  # use_vocab is false because we want Bert.

    def postprocess_to_id_and_replace_pad(batch, y):
        return [[replace_tokens(english_bert_tokenizer.convert_tokens_to_ids(token)) for token in sentence_list] for
                sentence_list in batch]

    trg_field = data.ReversibleField(init_token='<sos>',
                                     eos_token='<eos>',
                                     sequential=True)

    # preprocessing=english_bert_tokenizer.convert_tokens_to_ids,
    # postprocessing=postprocess_to_id_and_replace_pad)

    train_data, valid_data, test_data = Multi30k.splits(exts=(input_lang, output_lang),
                                                        fields=(src_field, trg_field))

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    # print(train_data.examples[0])

    trg_field.build_vocab(train_data, min_freq=2, vectors="glove.6B.100d")

    #     print(f"Unique tokens in source (", input_lang, ") vocabulary: ", len(SRC.vocab))
    # print(f"Unique tokens in target (", output_lang, ") vocabulary: ", len(trg_field.vocab))

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=gru_hyperparameters['batch_size'],
        device=fixed_vars['device'])

    return {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data,
            'train_iterator': train_iterator, 'valid_iterator': valid_iterator, 'test_iterator': test_iterator,
            'bert_tokenizer': multilingual_bert_tokenizer,
            'bert_encoder': bert_encoder, 'src_field': src_field, 'trg_field': trg_field}
