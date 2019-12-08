import pandas as pd
from global_variables import *
import matplotlib.pyplot as plt

path = os.path.join(fixed_vars['autoencoder_directory'], "loss.csv")
translator_loss_path = os.path.join(fixed_vars['translator_directory'])


def viz_loss(path, name=''):
    loss_df = pd.read_csv(path)
    # loss_df['epoch'] = loss_df['batch_num'].apply(lambda x: x * 40 / 29000)
    loss_df['rolling_loss'] = loss_df['validation_loss'].rolling(window=10).mean()
    loss_df['rolling_bleu'] = loss_df['bleu'].rolling(window=10).mean()
    # gca stands for 'get current axis'
    plt.clf()
    ax = plt.gca()
    # ax.set_ylim(0, 10)
    plt.title('BALM Translator Loss Curve' + name, fontsize=16)

    loss_df.plot(kind='line', x='batch_num', y='rolling_loss', ax=ax)
    # ax.autoscale(enable=True, axis="y", tight=False)

    plt.xlabel('Batch Number', fontsize=14)
    plt.ylabel('Cross-Entropy Validation Loss', fontsize=14)

    plt.show()

    plt.clf()
    ax = plt.gca()
    # ax.set_ylim(0, 10)
    plt.title('BALM Translator BLEU Curve' + name, fontsize=16)

    loss_df.plot(kind='line', x='batch_num', y='rolling_bleu', ax=ax)
    # ax.autoscale(enable=True, axis="y", tight=False)

    plt.xlabel('Batch Number', fontsize=14)
    plt.ylabel('BLEU', fontsize=14)

    plt.show()

    # good_batch = loss_df.loc[loss_df['rolling_loss'] < 0.01,]

    # print(name, good_batch.reset_index().loc[0, 'batch_num'])


# viz_loss(path)

for prop in [1]:
# for prop in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    viz_loss(os.path.join(translator_loss_path, str(prop) + 'loss.csv'), str(prop))
