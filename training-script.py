from autoencoder import Autoencoder
from transformer import *
import torch.optim as optim
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import math
import time
import random
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import logging

#  https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
#  https://github.com/huggingface/pytorch-pretrained-BERT#usage

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
gpu_bool = torch.cuda.device_count() > 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

print("loaded Bert Tokenizer")

# Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = bert_tokenizer.tokenize(text)

print(tokenized_text)

# Convert token to vocabulary indices
indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
bert_encoder = BertModel.from_pretrained('bert-base-cased')
bert_encoder.eval()

print("Loaded Bert")

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device)
bert_encoder.to(device)

print(tokens_tensor.size())
print(segments_tensors.size())
# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = bert_encoder(tokens_tensor)
    # encoded_layers, _ = bert_encoder(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12

print([l.size() for l in encoded_layers])  # we get 12 elements of 768 each


SRC = Field(tokenize=bert_tokenizer.tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize=bert_tokenizer.tokenize,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(SRC, TRG))

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

print(vars(train_data.examples[0]))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# SRC.build_vocab(train_data, vectors="glove.6B.100d")
# vocab = TEXT.vocab

print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 512
N_LAYERS = 3
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

FILTER_SIZE = 5
DECODER_HIDDEN_DIM = 3
shrink_net = ShrinkNet(bert_encoder.config.hidden_size, DECODER_HIDDEN_DIM)

print("made shrink-net!")

test_layer = DecoderLayer(DECODER_HIDDEN_DIM, FILTER_SIZE, DEC_DROPOUT)

def count_parameters(model):
    val = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(val)
    return val

count_parameters(bert_encoder)
count_parameters(shrink_net)
count_parameters(test_layer)

# enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
transformer_decoder = Decoder(DECODER_HIDDEN_DIM, bert_encoder.config.vocab_size, FILTER_SIZE, DEC_DROPOUT, N_LAYERS)

autoencoder = Autoencoder(bert_encoder, shrink_net, transformer_decoder, device).to(device)

print(autoencoder(tokens_tensor, tokens_tensor))
raise ValueError





print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(autoencoder.parameters())

PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
