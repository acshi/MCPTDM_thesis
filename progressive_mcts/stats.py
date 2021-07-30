#!/usr/bin/python3
import pdb
import numpy as np
import localreg
from matplotlib import pyplot as plt

data = np.genfromtxt("results.cache")
sum_repeated = data[:, 3]
data = data[sum_repeated > 0]

regret = data[:, 1] - data[:, 2]
sum_repeated = data[:, 3]
max_repeated = data[:, 4]
repeated_cost_avg = data[:, 5]

# pdb.set_trace()

# plt.scatter(sum_repeated, regret)
# plt.show()
# plt.scatter(max_repeated, regret)
# plt.show()

plt.scatter(repeated_cost_avg, regret)
# reg_ys = localreg.localreg(
#     repeated_cost_avg, regret, degree=0, kernel=localreg.rbf.gaussian, width=0.01)
# plt.scatter(repeated_cost_avg, reg_ys)
plt.show()
