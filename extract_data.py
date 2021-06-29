import glob
import gzip
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# find all files with data
file_list = glob.glob("data/cases_*.csv.gz")
random_files = random.sample(file_list, 20)

# read each data batch and append it to a list of batches
batches = []
for i, x in enumerate(random_files):
    print(f"Reading file {i+1}/{len(random_files)}...")
    try:
        y = np.loadtxt(gzip.GzipFile(x, "r"), skiprows=1, usecols=(7, 24), delimiter=",", dtype=str)
        batches.append(y)
    except:
        print(f"Couldn't read file: {x}")

# throw together all batches in a single list
cases = np.array(batches).reshape(-1, 2)

# clean data: keep only data points for which we have an age and for which the outcome
# is either "Recovered" or "Death"
cases = [x for x in cases if x[0] != "" and x[1] in ("Recovered", "Death")]

# convert age to integer number and recode "Death" as 0 and "Recovered" as 1
cases = np.array([[int(float(x[0])), 1 if x[1] == "Recovered" else 0] for x in cases])

# save data for later use
np.save("data.npy", cases)

# visualize data: plot distributions of ages and outcomes in the population sample and  
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.hist(cases[:,0])
ax1.set_xlabel("age")
ax1.set_ylabel("count")

ax2.bar((0, 2), (cases[:,1].sum(), len(cases) - cases[:,1].sum()))
ax2.set_xticks((0, 2))
ax2.set_xticklabels(("recovered", "death"))
ax2.set_ylabel("count")

fig.tight_layout()

plt.show()
