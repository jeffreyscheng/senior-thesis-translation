import pandas as pd
from global_variables import *
import matplotlib.pyplot as plt

loss_df = pd.read_csv(os.path.join(fixed_vars['autoencoder_directory'], "loss.csv"))
loss_df['rolling_loss'] = loss_df['loss'].rolling(window=1000).mean()
# gca stands for 'get current axis'
ax = plt.gca()
loss_df.plot(kind='line', x='batch_num', y='loss', ax=ax)
plt.show()
