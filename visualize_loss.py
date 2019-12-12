import pandas as pd
from global_variables import *
import matplotlib.pyplot as plt

path = os.path.join(fixed_vars['autoencoder_directory'], "loss.csv")
translator_loss_path = os.path.join(fixed_vars['translator_directory'])


def viz_loss(path, name='', bleu=True):
    loss_df = pd.read_csv(path)
    # loss_df['epoch'] = loss_df['batch_num'].apply(lambda x: x * 40 / 29000)
    loss_df['rolling_loss'] = loss_df['validation_loss'].rolling(window=10).mean()
    # gca stands for 'get current axis'
    plt.clf()
    ax = plt.gca()
    # ax.set_ylim(0, 10)
    plt.title(name + ' Loss Curve', fontsize=16)

    loss_df.plot(kind='line', x='batch_num', y='rolling_loss', ax=ax)
    # ax.autoscale(enable=True, axis="y", tight=False)

    plt.xlabel('Batch Number', fontsize=14)
    plt.ylabel('Cross-Entropy Validation Loss', fontsize=14)

    plt.show()

    if bleu:
        loss_df['rolling_bleu'] = loss_df['bleu'].rolling(window=10).mean()
        plt.clf()
        ax = plt.gca()
        # ax.set_ylim(0, 10)
        plt.title(name + ' BLEU Curve', fontsize=16)

        loss_df.plot(kind='line', x='batch_num', y='rolling_bleu', ax=ax)
        # ax.autoscale(enable=True, axis="y", tight=False)

        plt.xlabel('Batch Number', fontsize=14)
        plt.ylabel('BLEU', fontsize=14)

        plt.show()

    # good_batch = loss_df.loc[loss_df['rolling_loss'] < 0.01,]

    # print(name, good_batch.reset_index().loc[0, 'batch_num'])


for model in range(2):
    autoencoder_path = os.path.join(fixed_vars['root_directory'], 'autoencoder-' + str(model), 'loss.csv')
    viz_loss(autoencoder_path, 'BALM Autoencoder ' + str(model), bleu=True)


for prop in [1]:
# for prop in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    viz_loss(os.path.join(translator_loss_path, str(prop) + 'loss.csv'), str(prop))
