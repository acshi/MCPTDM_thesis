#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pdb

data = np.genfromtxt("all_mac_report")

plt.hist(data)
plt.ylim([0, 1000])
plt.show()
