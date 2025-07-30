import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Prepare data (replace commas with dots and convert to float)
data = {
    'Training method': ['Supervised', 'Unsupervised', 'No grad', 'Supervised', 'Unsupervised', 'No grad'],
    'Input': ['pred score', 'zero', 'zero', 'zero', 'pred score', 'pred score'],
    'Score all': [65.1615, 50.7243, 59.3115, 60.0857, 1, 59.8603],
    'Score close': [91.9761, 82.5275, 88.1436, 89.4779, 1, 86.953],
    'Score far': [30.2833, 14.0036, 25.9004, 23.4772, 1, 26.0382]
}

df = pd.DataFrame(data)

# Aggregate by Training method and Input (average scores)
df_grouped = df.groupby(['Training method', 'Input']).mean().reset_index()

methods = df_grouped['Training method'].unique()
inputs = df_grouped['Input'].unique()

x = np.arange(len(methods))  # the label locations
width = 0.35  # the width of the bars

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

score_types = ['Score all', 'Score close', 'Score far']

for i, score in enumerate(score_types):
    ax = axes[i]
    
    # Values for each input type
    vals_input0 = []
    vals_input1 = []
    
    for method in methods:
        vals_input0.append(df_grouped[(df_grouped['Training method'] == method) & (df_grouped['Input'] == inputs[0])][score].values[0])
        vals_input1.append(df_grouped[(df_grouped['Training method'] == method) & (df_grouped['Input'] == inputs[1])][score].values[0])
    
    rects1 = ax.bar(x - width/2, vals_input0, width, label=inputs[0])
    rects2 = ax.bar(x + width/2, vals_input1, width, label=inputs[1])
    
    ax.set_title(score)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_xlabel('Training Method')
    if i == 0:
        ax.set_ylabel('Score')
    ax.legend(title='Input')

plt.suptitle('Scores by Training Method and Input')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('scores_histogram.png', dpi=300)
