import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def draw_plane(theta_val, theta_0):
    slope = -1 / (theta_val[1] / theta_val[0])
    x1 = 1
    y1 = slope * x1 + theta_0
    cx = -3
    cy = -3
    r_sq = 25

    figure, axes = plt.subplots()
    Drawing_colored_circle = plt.Circle((cx, cy), r_sq ** 0.5, fill=False)

    axes.set_aspect(1)
    axes.add_artist(Drawing_colored_circle)
    plt.scatter(x[:, 0], x[:, 1])
    plt.plot([0, x1, -1], [0, y1, -1 * slope + theta_0])
    plt.show()



x = np.arange(7)
y = np.arange(7)
plt.plot(x, y)
plt.show()