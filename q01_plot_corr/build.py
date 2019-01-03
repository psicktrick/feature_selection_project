# %load q01_plot_corr/build.py
# Default imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
plt.switch_backend('agg')
import seaborn as sns
data = pd.read_csv('data/house_prices_multivariate.csv')


# Write your solution here:
def plot_corr(data, size=11):
    sns.heatmap(data.corr(), cmap='YlOrRd')
    plt.show()
plot_corr(data, size=11)
sns.heatmap(data.corr(), cmap='YlOrRd')
plt.show()
data.head()
data.dtypes


