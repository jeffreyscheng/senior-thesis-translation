import pandas as pd
from global_variables import *
import matplotlib.pyplot as plt

loss_df = pd.read_csv(os.path.join(fixed_vars['root_directory'], "loss.csv"))

# gca stands for 'get current axis'
ax = plt.gca()
loss_df.plot(kind='line', x='batch_num', y='loss', ax=ax)
plt.show()
