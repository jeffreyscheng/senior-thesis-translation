from nltk.translate.bleu_score import sentence_bleu
import torch
import os
import pandas as pd
import time
from src.global_variables import fixed_vars


class Pipeline:
    def __init__(self, model_path, creation_fn, hyperparameters):
        self.model_path = model_path
        self.creation_fn = creation_fn
        self.hyperparameters = hyperparameters
        self.model, self.optimizer, self.loss_df = self.read_state_from_path()

    @staticmethod
    def get_bleu(reference, output):
        return sentence_bleu([reference], output)  # TODO: add case for batch?

    def write_state_to_path(self):
        state = {'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),
                 'loss_df': self.loss_df,
                 'hyperparameters': self.hyperparameters}
        torch.save(state, os.path.join(self.model_path, 'state.pkl'))

    def read_state_from_path(self):
        prev_state_exists = os.path.isfile(os.path.join(self.model_path, 'state.pkl'))
        model, optimizer = self.creation_fn(self.hyperparameters)
        loss_df = pd.DataFrame(columns=['batch_num', 'validation_loss', 'bleu'])
        if not self.hyperparameters['restart'] and prev_state_exists:
            state = torch.load(os.path.join(self.model_path, 'state.pkl'))
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            loss_df = state['loss_df']
            print("Loaded previous state!")
        elif self.hyperparameters['restart'] and prev_state_exists:
            print("Will overwrite previous state with fresh model!")
        elif not prev_state_exists:
            print("No previous state found; initializing fresh model!")
        return model, optimizer, loss_df

    def token_ids_to_token_ids(self, x):
        output = self.model(x, None, 0)
        output = output[1:]
        _, best_guess = torch.max(output, dim=2)
        return best_guess

    def get_batch_loss(self, batch, criterion):
        input_batch = batch.src.to(fixed_vars['device'])
        reference_batch = batch.trg.to(fixed_vars['device'])
        self.optimizer.zero_grad()
        output = self.model(input_batch, reference_batch)
        output = output[1:].view(-1, output.shape[-1])
        reference_batch = reference_batch[1:].view(-1)
        loss = criterion(output, reference_batch)
        del input_batch, reference_batch, output
        return loss

    def token_ids_to_token_list(self, x):
        pass

    def train_translator(self, translation_objects, criterion, clip, num_epochs):
        rows = []
        total_num_batches = len(translation_objects['train_data']) * num_epochs / self.hyperparameters['batch_size']
        while True:
            if self.model.number_of_batches_seen > total_num_batches:
                break
            tick = time.time()
            for i, batch in enumerate(translation_objects['train_iterator']):
                if self.model.number_of_batches_seen > total_num_batches:
                    break
                loss = self.get_batch_loss(batch, criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                del loss
            print("Finished training:", time.time() - tick)

            tick = time.time()
            losses = []
            with torch.no_grad():
                for i, batch in enumerate(translation_objects['valid_iterator']):
                    loss = self.get_batch_loss(batch, criterion)
                    losses.append(loss.data)

                validation_loss = float(sum(losses) / len(losses))

                # stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
                bleu_score_list = []
                for i, batch in enumerate(translation_objects['test_iterator']):
                    input_batch = batch.src.to(fixed_vars['device'])
                    output_ids = self.token_ids_to_token_ids(input_batch)
                    best_guess = self.token_ids_to_token_list(output_ids)
                    reference_batch = batch.trg.to(fixed_vars['device'])
                    ref_tokens = self.token_ids_to_token_list(reference_batch)
                    bleu_score_list.append(self.get_bleu(ref_tokens, best_guess))
                bleu_score = sum(bleu_score_list) / len(bleu_score_list)
                rows.append({'batch_num': self.model.number_of_batches_seen, 'validation_loss': validation_loss,
                             'bleu': bleu_score})

                print('Finished validating: ', time.time() - tick, ', Validation loss: ', validation_loss, "BlEU:",
                      bleu_score)

            self.loss_df = self.loss_df.append(pd.DataFrame(rows), ignore_index=True)
            self.write_state_to_path()
        return self.model, self.loss_df


class EnglishAutoencoder(Pipeline):
    def __init__(self, english_tokenizer, english_field, model_path, creation_fn, hyperparameters):
        Pipeline.__init__(self, model_path, creation_fn, hyperparameters)
        self.english_tokenizer = english_tokenizer
        self.english_field = english_field

    def sentence_to_sentence(self, x):
        input_tokens = self.english_tokenizer.tokenize(x)
        output_tokens = self.token_list_to_token_list(input_tokens)
        return ''.join(output_tokens)

    def token_list_to_token_list(self, x):
        input_english_ids = self.english_tokenizer.convert_tokens_to_ids(x)
        formatted_english_ids = self.english_field.preprocess(input_english_ids)  # should add the CLS, UNK, EOS tokens
        output_english_ids = self.token_ids_to_token_ids(formatted_english_ids)
        return self.english_tokenizer.convert_ids_to_tokens(output_english_ids)

    def token_ids_to_token_list(self, x):
        try:
            return self.english_tokenizer.convert_ids_to_tokens(x)
        except:
            return self.english_tokenizer.convert_ids_to_tokens(x.tolist())


class GermanToEnglishTranslator(Pipeline):
    def __init__(self, english_tokenizer, german_tokenizer, english_field, german_field, model_path, creation_fn,
                 hyperparameters):
        Pipeline.__init__(self, model_path, creation_fn, hyperparameters)
        self.english_tokenizer = english_tokenizer
        self.german_tokenizer = german_tokenizer
        self.english_field = english_field
        self.german_field = german_field

    def sentence_to_sentence(self, x):
        input_tokens = self.german_tokenizer.tokenize(x)
        output_tokens = self.token_list_to_token_list(input_tokens)
        return ''.join(output_tokens)

    def token_list_to_token_list(self, x):
        input_german_ids = self.german_tokenizer.convert_tokens_to_ids(x)
        formatted_german_ids = self.german_field.preprocess(input_german_ids)  # should add the CLS, UNK, EOS tokens
        output_english_ids = self.token_ids_to_token_ids(formatted_german_ids)
        return self.english_tokenizer.convert_ids_to_tokens(output_english_ids)

    def token_ids_to_token_list(self, x):
        try:
            return self.english_tokenizer.convert_ids_to_tokens(x)
        except:
            return self.english_tokenizer.convert_ids_to_tokens(x.tolist())