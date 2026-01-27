import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
print(df.head())
print(df.shape)
print(df['label'].value_counts())
# Any NaNs?
print(df.isnull().sum())

# Any unexpected labels?
print(sorted(df['label'].unique()))


def plot_hand(row):
    x = row[[f'x{i}' for i in range(21)]].values
    y = row[[f'y{i}' for i in range(21)]].values
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y)
    for i in range(21):
        plt.text(x[i], y[i], str(i))
    plt.gca().invert_yaxis()
    plt.title(f"Label: {row['label']}")
    plt.savefig(f"{row['label']}_sample.png")
    plt.close()
# Plot one sample per letter
for label in df['label'].unique():
    sample = df[df['label'] == label].sample(1).iloc[0]
    plot_hand(sample)
