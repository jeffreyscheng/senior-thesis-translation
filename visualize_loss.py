import pandas as pd
from global_variables import *
import matplotlib.pyplot as plt

# path = os.path.join(fixed_vars['autoencoder_directory'], "loss.csv")
path = os.path.join(fixed_vars['translator_directory'], "loss.csv")


def viz_loss(path):
    loss_df = pd.read_csv(path)
    loss_df['epoch'] = loss_df['batch_num'].apply(lambda x: x * 40 / 29000)
    loss_df['rolling_loss'] = loss_df['loss'].rolling(window=10).mean()
    # gca stands for 'get current axis'
    plt.clf()
    ax = plt.gca()
    plt.title('English Sentence Autoencoder Learning Curve', fontsize=16)

    loss_df.plot(kind='line', x='epoch', y='rolling_loss', ax=ax)

    plt.xlabel('Epoch Number', fontsize=14)
    plt.ylabel('Cross-Entropy Training Loss', fontsize=14)

    plt.show()


viz_loss(path)
