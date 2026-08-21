import numpy as np
import matplotlib.pyplot as plt

starting_x = 100
cutting_rate = 1
control_width = 64//cutting_rate
control_length = 128//cutting_rate

nb_coeffs = 1

x = np.arange(starting_x, starting_x + control_width, 1)
y = np.arange(0, control_length, 1)
X, Y = np.meshgrid(x, y)

########################## Hand made W distribution
# x_grid = starting_x + np.arange(0.0, control_width, 1)*cutting_rate
# y_grid = np.arange(0.0, control_length, 1)*cutting_rate
# k_modes = np.arange(1, nb_coeffs + 1, dtype=np.float32)
# sin_x = np.sin(2*np.pi/(control_width * cutting_rate) * k_modes[:, None] * x_grid[None, :])
# sin_y = np.sin(2*np.pi/(control_length * cutting_rate) * k_modes[:, None] * y_grid[None, :])

# a = [[np.random.randn() for _ in range(nb_coeffs)]]
# c = [-0.0623]

# W = np.zeros((control_width, control_length))
# for i in range(nb_coeffs):
#     W += c[i] * sin_x[i][:, None] * sin_y[i][None, :]

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# surface = ax.plot_surface(X, Y, W.T, cmap='viridis', edgecolor='none')

# ax.set_title("Blowing Velocity W Distribution Wanted, Handmade")
# ax.set_xlabel("Position X")
# ax.set_ylabel("Position Y")
# ax.set_zlabel("Velocity W")
# fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

# plt.show()

########################## W distribution by RL model

data = np.loadtxt('single_grid_input.txt')

W = np.zeros((control_width, control_length))
for x, y, w in data:
    int_x = int(x)
    int_y = int(y)
    W[int_x-starting_x, int_y] = w

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(X, Y, W.T, cmap='viridis', edgecolor='none')

ax.set_title("Blowing Velocity W Distribution Simulated by RL Model")
ax.set_xlabel("Position X")
ax.set_ylabel("Position Y")
ax.set_zlabel("Velocity W")
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

plt.show()
