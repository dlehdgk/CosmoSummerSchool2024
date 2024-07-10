# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import glob


# Function to read data from a single file
def read_data(filename):
    with open(filename, "rb") as fp:
        nx = np.frombuffer(fp.read(4), dtype=np.int32)[0]
        ny = np.frombuffer(fp.read(4), dtype=np.int32)[0]
        den = np.frombuffer(fp.read(4 * nx * ny), dtype=np.float32)
    return nx, ny, den.reshape((nx, ny))


# Use glob to find all files matching the pattern
file_pattern = "Run/xzslice.*"  # Adjust the pattern as needed
files = sorted(glob.glob(file_pattern))
# Read data from all files
frames = []
for filename in files:
    nx, ny, data = read_data(filename)
    frames.append(data)

# Redshift values starting from 47 to present
redshift = np.arange(47, -0.1, -0.1)

# Create a figure and axis
fig, ax = plt.subplots()
cax = ax.matshow(frames[0], cmap="jet", origin="lower")
ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)


# Function to update the plot for each frame
def update(i):
    cax.set_data(frames[i])
    ax.set_title(r"z=%.1f" % (redshift[i]))
    return (cax,)


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)

# Display the animation
plt.colorbar(cax)
plt.xlabel("z [$h^{-1}$ Mpc]")
plt.ylabel("x [$h^{-1}$ Mpc]")


FFwriter = animation.FFMpegWriter(fps=30)
ani.save("Outputs/evolution.mp4", writer=FFwriter)
# ani.save("web.gif", writer="ffmpeg", fps=30)
