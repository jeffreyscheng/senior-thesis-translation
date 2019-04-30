import pandas as pd
from global_variables import *
import matplotlib.pyplot as plt
from models import Autoencoder
from set_up_translation import get_translation_objects

loss_df = pd.read_csv(os.path.join(fixed_vars['root_directory'], "gru-8", "loss.csv"))
loss_df['rolling_loss'] = loss_df['loss'].rolling(window=1000).mean()
# gca stands for 'get current axis'
ax = plt.gca()
loss_df.plot(kind='line', x='batch_num', y='rolling_loss', ax=ax)
plt.show()
