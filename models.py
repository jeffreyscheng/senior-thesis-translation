import torch
import torch.nn as nn
import random


class GRUDecoder(nn.Module):
    def __init__(self, emb_dim, vocab, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim  # should be 1 if we don't embed
        self.vocab_size = len(vocab)
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)  # we used GLove 100-dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers  # should be 1
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(self.hid_dim, self.vocab_size)

    def forward(self, last_word, last_hidden):
        output, new_hidden = self.rnn(last_word.float(), last_hidden.float())
        prediction = self.out(output.squeeze(0))
        return prediction, new_hidden


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        # self.shrink = shrink_net
        self.decoder = decoder
        self.device = device
        self.number_of_batches_seen = 0

    # only purpose is to train encoder and decoder; doesn't need one without a target
    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = trg.shape[1]
        if trg is None:
            max_len = 100
        else:
            max_len = trg.shape[0]

        outputs = torch.zeros(max_len, batch_size, self.decoder.vocab_size).to(self.device)

        src = src.permute(1, 0)
        hidden = self.encoder(src)

        #  https://huggingface.co/transformers/model_doc/bert.html
        hidden = hidden[0]  # ignore pooled output
        hidden = torch.mean(hidden, dim=1)  # get sentence embedding from mean of word embeddings
        hidden = hidden.unsqueeze(dim=0)

        # first input to the decoder is the <sos> tokens
        curr_token = trg[0, :]

        for t in range(1, max_len):
            curr_token = self.decoder.embed(curr_token)
            curr_token = curr_token.unsqueeze(dim=0)
            new_output, hidden = self.decoder(curr_token, hidden)
            outputs[t] = new_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = new_output.max(1)[1]
            curr_token = (trg[t, :] if teacher_force else top1)

        self.number_of_batches_seen += 1
        return outputs


class Translator(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 768)
        self.decoder = decoder
        self.device = device
        self.number_of_batches_seen = 0
        self.relu = nn.ReLU(True)

    # only purpose is to train encoder and decoder; doesn't need one without a target
    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = trg.shape[1]
        if trg is None:
            max_len = 100
        else:
            max_len = trg.shape[0]

        outputs = torch.zeros(max_len, batch_size, self.decoder.vocab_size).to(self.device)

        src = src.permute(1, 0)
        german_thought = self.encoder(src)

        #  https://github.com/huggingface/pytorch-pretrained-BERT#usage
        german_thought = german_thought[0]  # ignore pooled output
        german_thought = torch.mean(german_thought, dim=1)  # get sentence embedding from mean of word embeddings

        german_thought = self.fc1(german_thought)
        german_thought = self.relu(german_thought)
        german_thought = self.fc2(german_thought)
        english_thought = self.relu(german_thought)

        english_thought = english_thought.unsqueeze(dim=0)
        hidden = english_thought

        # first input to the decoder is the <sos> tokens
        curr_token = trg[0, :]

        for t in range(1, max_len):
            curr_token = self.decoder.embed(curr_token)
            curr_token = curr_token.unsqueeze(dim=0)
            new_output, hidden = self.decoder(curr_token, hidden)
            outputs[t] = new_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = new_output.max(1)[1]
            curr_token = (trg[t, :] if teacher_force else top1)

        self.number_of_batches_seen += 1
        return outputs


def count_parameters(model):
    val = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(val)
    return val
