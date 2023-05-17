import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

f = h5.File('../../job/test/output/test_force.h5', 'r')
force = f['fx']
print(force.shape)
r = np.linspace(0, 1, 32)  # Radius
t = np.linspace(0, 2*np.pi, 128)  # Theta

R, T = np.meshgrid(r, t)  # This creates a 2D grid of r, t values

for i in range(100):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},dpi=100)
    print(i)
    # Plot the data
    c = ax.pcolormesh(T, R, force[i,:,:,0], cmap='viridis')
    # plt.colorbar(c)
    fig.colorbar(c, ax=ax)
    ax.set_theta_zero_location("N")

    # Show the plot
    plt.savefig('fig/fx_'+str(i)+'.png')
    plt.close