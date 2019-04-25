from global_variables import *
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

multilingual_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
english_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
bert_encoder.to(fixed_vars['device'])
bert_encoder.eval()

input_lang = '.en'
output_lang = '.en'


def replace_tokens(x):
    if x == "<sos>":
        return 28993
    elif x == '<pad>':
        return 28994
    elif x == '<eos>':
        return 28995
    else:
        return x


def postprocess_replace_pad(batch, y):
    return [[replace_tokens(token) for token in sentence_list] for sentence_list in batch]


def postprocess_to_id_and_replace_pad(batch, y):
    return [[replace_tokens(english_bert_tokenizer.convert_tokens_to_ids(token)) for token in sentence_list] for sentence_list in batch]


# originally had <sos> and <eos>
src_field = Field(tokenize=multilingual_bert_tokenizer.tokenize,
                  init_token='<sos>',
                  eos_token='<eos>',
                  preprocessing=multilingual_bert_tokenizer.convert_tokens_to_ids,
                  postprocessing=postprocess_replace_pad,
                  use_vocab=False)  # use_vocab is false because we want Bert.
trg_field = Field(tokenize=english_bert_tokenizer.tokenize,
                  init_token='<sos>',
                  eos_token='<eos>',
                  # preprocessing=english_bert_tokenizer.convert_tokens_to_ids,
                  postprocessing=postprocess_to_id_and_replace_pad)

train_data, valid_data, test_data = Multi30k.splits(exts=(input_lang, output_lang),
                                                    fields=(src_field, trg_field))

print(len(english_bert_tokenizer.vocab))
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

print(train_data.examples[0].__dict__)

trg_field.build_vocab(train_data, min_freq=2, vectors="glove.6B.100d")
# print("Length of English vocabulary: ", len(trg_field.vocab))
# print(vars(train_data.examples[0]))

# print(trg_field.__dict__)

#     print(f"Unique tokens in source (", input_lang, ") vocabulary: ", len(SRC.vocab))
# print(f"Unique tokens in target (", output_lang, ") vocabulary: ", len(trg_field.vocab))

# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data),
#     batch_size=gru_hyperparameters['batch_size'],
#     device=fixed_vars['device'])
