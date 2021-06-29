from collections import Counter
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def equalize(dataset):
    recovered_indices = dataset[dataset['recovered'] == 1].index.tolist()
    death_indices = dataset[dataset['recovered'] == 0].index.tolist()
    recovered_indices = sample(recovered_indices, len(death_indices))
    equalized = dataset.loc[recovered_indices + death_indices]

    return equalized


def plot_data(dataset, ax):
    ax.scatter(dataset['age'], dataset['recovered'], alpha=0.1)
    ax.set_yticks((0, 1))
    ax.set_yticklabels(('death', 'recovered'))
    ax.set_xlabel("age [years]")


# load dataset
age_recovered = np.load("data.npy")
age_recovered = pd.DataFrame.from_dict({"age": age_recovered[:,0], "recovered": age_recovered[:,1]})

# plot data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
subsampled = age_recovered.loc[sample(age_recovered.index.tolist(), 5000)]
equalized = equalize(subsampled)
plot_data(subsampled, ax1)
plot_data(equalized, ax2)
ax1.set_title("imbalanced dataset")
ax2.set_title("balanced dataset")
fig.tight_layout()
plt.show()

input("Press Enter to continue...")

# split in training- and test set
age_recovered_copy = age_recovered.copy()
training_set = age_recovered_copy.sample(frac=0.75, random_state=0)

test_set = age_recovered_copy.drop(training_set.index)

# calculate sample weights
counter = Counter(training_set['recovered'])
counts = dict(counter.most_common())
total = len(training_set)
weights = np.array([1.0 / counts[recovered]
                    for recovered in training_set['recovered']])
weights /= weights.sum()

# fit logistic regression model
log_reg = LogisticRegression()
log_reg = log_reg.fit(training_set['age'].values.reshape(-1,1),
                      training_set['recovered'].values,
                      weights)

input("Press Enter to continue...")

# predict values from test set
predicted_recovered = log_reg.predict(test_set['age'].values.reshape(-1,1))

# assess accuracy
print("Balanced accuracy:",
      balanced_accuracy_score(test_set['recovered'].values,
                              predicted_recovered))

# plot data and survival probability
fig, ax = plt.subplots()
plot_data(subsampled, ax)
all_ages = np.linspace(0, 120, 120).reshape(-1, 1)
ax.plot(log_reg.predict_proba(all_ages)[:,1],
        label="survival probability")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()

input("Press Enter to continue...")

# predict survival for arbitrary age
def predict(age):
    survival_boolean = log_reg.predict(np.array([age]).reshape(-1, 1))[0]
    print(f"A person aged {age} will likely",
          "survive the infection" if survival_boolean  else "not survive the infection")
    
predict(89)
predict(22)
