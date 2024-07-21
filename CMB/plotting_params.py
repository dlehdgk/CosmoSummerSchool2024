# %%
from average_finder import *
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

average = stack_cmb_params(20000)

# %%
step = 8
fig, ax = plt.subplots(5, 2, figsize=(16, 24), dpi=300)
for minmax in range(2):
    x, y, u, v = vectormap(step, average[minmax, 1, :, :], average[minmax, 2, :, :])
    x, y, ur, vr = vectormap(step, average[minmax, 3, :, :], average[minmax, 4, :, :])
    for params, name in zip(
        range(5),
        [
            "Temperature",
            "Q Polarisation",
            "U Polarisation",
            "Qr Polarisation",
            "Ur Polarisation",
        ],
    ):
        # plotting all averages
        im = ax[params, minmax].imshow(
            average[minmax, params, :, :], extent=(-2.5, 2.5, -2.5, 2.5), cmap="jet"
        )
        ax[params, minmax].grid()
        ax[params, minmax].set_xlabel("Degrees")
        ax[params, minmax].set_ylabel("Degrees")
        if minmax == 0:
            ax[params, minmax].set_title(r"Average of the maxima of %s" % name)
        else:
            ax[params, minmax].set_title(r"Average of the minima of %s" % name)
        divider = make_axes_locatable(ax[params, minmax])
        cax = divider.append_axes("right", size="20%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format="%.2f")
        if params > 0 and params < 3:
            ax[params, minmax].quiver(x, y, u, v, scale=15, headwidth=1, color="black")
        elif params >= 3:
            ax[params, minmax].quiver(
                x, y, ur, vr, scale=15, headwidth=1, color="black"
            )
fig.savefig("Output/Average.png")
fig.show()
