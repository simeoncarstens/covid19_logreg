import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def plot_data(dataset, ax):
    ax.scatter(dataset['age'],
               dataset['death'], label="")
    ax.set_yticks((0, 1))
    ax.set_yticklabels(('alive', 'dead'))
    ax.set_xlabel("age [years]")


# load dataset
filename = "COVID19_line_list_data.csv"
dataset = pd.read_csv(filename)

# select relevant data
age_death = dataset[['age', 'death']]

# filter dataset for default / invalid values (part of "cleaning")
age_death_clean = age_death.dropna()
age_death_clean = age_death_clean[age_death_clean['death'].isin(('0', '1'))]
age_death_clean['death'] = pd.to_numeric(age_death_clean['death'])

# plot data
fig, ax = plt.subplots()
plot_data(age_death_clean, ax)
plt.show()

# split in training- and test set
age_death_clean_copy = age_death_clean.copy()
training_set = age_death_clean_copy.sample(frac=0.75, random_state=0)
test_set = age_death_clean_copy.drop(training_set.index)

# calculate sample weights
counter = Counter(training_set['death'])
counts = dict(counter.most_common())
total = len(training_set)
weights = np.array([1.0 / counts[death]
                    for death in training_set['death']])
weights /= weights.sum()

# fit logistic regression model
log_reg = LogisticRegression()
log_reg = log_reg.fit(training_set['age'].values.reshape(-1,1),
                      training_set['death'].values,
                      weights)

# predict values from test set
predicted_death = log_reg.predict(test_set['age'].values.reshape(-1,1))

# assess accuracy
print("Balanced accuracy:",
      balanced_accuracy_score(test_set['death'].values,
                              predicted_death))

# plot data and survival probability
fig, ax = plt.subplots()
plot_data(age_death_clean, ax)
all_ages = np.linspace(0, 100, 100).reshape(-1, 1)
ax.plot(log_reg.predict_proba(all_ages)[:,1],
        label="survival probability")
ax.legend(frameon=False)
plt.show()
