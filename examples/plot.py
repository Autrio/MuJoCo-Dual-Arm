import numpy as np
import matplotlib.pyplot as plt

loss = np.load("./examples/loss.npy")

plt.figure()
plt.plot(loss)
plt.show()