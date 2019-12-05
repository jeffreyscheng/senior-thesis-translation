import pandas as pd
from global_variables import *
import matplotlib.pyplot as plt

path = os.path.join(fixed_vars['autoencoder_directory'], "loss.csv")
translator_loss_path = os.path.join(fixed_vars['translator_directory'])


def viz_loss(path, name=''):
    loss_df = pd.read_csv(path)
    # loss_df['epoch'] = loss_df['batch_num'].apply(lambda x: x * 40 / 29000)
    loss_df['rolling_loss'] = loss_df['loss'].rolling(window=10).mean()
    # gca stands for 'get current axis'
    plt.clf()
    ax = plt.gca()
    plt.title('BALM Translator Learning Curve' + name, fontsize=16)

    loss_df.plot(kind='line', x='batch_num', y='rolling_loss', ax=ax)

    plt.xlabel('Batch Number', fontsize=14)
    plt.ylabel('Cross-Entropy Training Loss', fontsize=14)

    plt.show()

    good_batch = loss_df.loc[loss_df['rolling_loss'] < 0.01,]

    print(name, good_batch.reset_index().loc[0, 'batch_num'])


# viz_loss(path)

for theta in range(1, 11):
    prop = theta / 20
    viz_loss(os.path.join(translator_loss_path, str(prop) + 'loss.csv'), str(prop))
