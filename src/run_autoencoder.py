from src.global_variables import fixed_vars, autoencoder_hyperparameters
from src.models import Autoencoder, GRUDecoder
import torch.optim as optim
from src.set_up_translation import autoencoder_objects, pad_index
from transformers import BertModel, RobertaModel
from src.pipelines import EnglishAutoencoder
import torch


def create_autoencoder_given_encoder(bert_encoder, vocab, hyperparameters):
    bert_encoder.to(fixed_vars['device'])
    bert_encoder.eval()
    gru_decoder = GRUDecoder(fixed_vars['word_embedding_dim'],
                             vocab,
                             fixed_vars['bert_embedding_dim'],
                             hyperparameters['gru_layers'],
                             hyperparameters['gru_dropout'])
    autoencoder = Autoencoder(bert_encoder,
                              gru_decoder,
                              fixed_vars['device']).to(fixed_vars['device'])
    return autoencoder


def create_roberta_autoencoder(hyperparameters):
    bert_encoder = RobertaModel.from_pretrained('roberta-base')
    vocab = autoencoder_objects['english_bert_tokenizer'].encoder
    autoencoder = create_autoencoder_given_encoder(bert_encoder, vocab, hyperparameters)
    autoencoder_optimizer = optim.Adam(autoencoder.decoder.parameters(),
                                       lr=hyperparameters['learning_rate'],
                                       weight_decay=10 ** (-5))
    return autoencoder, autoencoder_optimizer


def create_bert_autoencoder(hyperparameters):
    bert_encoder = BertModel.from_pretrained('bert-base-cased')
    vocab = autoencoder_objects['english_bert_tokenizer'].vocab
    autoencoder = create_autoencoder_given_encoder(bert_encoder, vocab, hyperparameters)
    autoencoder_optimizer = optim.Adam(autoencoder.decoder.parameters(),
                                       lr=hyperparameters['learning_rate'],
                                       weight_decay=10 ** (-5))
    return autoencoder, autoencoder_optimizer


if autoencoder_hyperparameters['roberta']:
    creation_fn = create_roberta_autoencoder
else:
    creation_fn = create_bert_autoencoder
autoencoder_pipeline = EnglishAutoencoder(english_tokenizer=autoencoder_objects['english_bert_tokenizer'],
                                          english_field=autoencoder_objects['english_field'],
                                          model_path=fixed_vars['autoencoder_directory'],
                                          creation_fn=creation_fn,
                                          hyperparameters=autoencoder_hyperparameters)

autoencoder_pipeline.train_translator(translation_objects=autoencoder_objects,
                                      criterion=torch.nn.CrossEntropyLoss(ignore_index=pad_index),
                                      clip=fixed_vars['gradient_clip'],
                                      num_epochs=autoencoder_hyperparameters['num_epochs'])
