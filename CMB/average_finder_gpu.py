import numpy as np
import healpy as hp
import camb
import torch
import multiprocessing as mp
from camb import model, initialpower
from joblib import Parallel, delayed, parallel_backend
from numba import jit
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

# Ensure we are using the MPS backend for PyTorch
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Define Qr and Ur parameters using Numba
@jit(nopython=True)
def Qr(Q, U, phi):
    return -Q * np.cos(2 * phi) - U * np.sin(2 * phi)


@jit(nopython=True)
def Ur(Q, U, phi):
    return Q * np.sin(2 * phi) - U * np.cos(2 * phi)


# Convert longitude and latitude to angle from east using Numba
@jit(nopython=True)
def east_phi(lon_c, lat_c, lon_p, lat_p):
    dlon = np.radians(lon_p - lon_c)
    dlat = np.radians(lat_p - lat_c)
    return np.arctan2(dlat, -dlon)


# Generate polarization vectors using Numba
@jit(nopython=True)
def pol_vec(Q, U):
    psi = 0.5 * np.arctan2(U, Q)
    P = np.sqrt(Q**2 + U**2)
    return psi, P


# Stack CMB parameters function
def stack_cmb_params(no_spots, param_file="planck_2018.ini", nside=512):
    # Use input initial params to generate CMB spectra
    planck2018pars = camb.read_ini(param_file)
    planck2018 = camb.get_results(planck2018pars)
    powers = planck2018.get_cmb_power_spectra(
        planck2018pars, CMB_unit="muK", raw_cl=True
    )
    aCl_Total = powers["total"]
    lmax = aCl_Total.shape[0] - 1
    almT, almE, almB = hp.synalm(
        [
            np.array(aCl_Total[:, 0]),
            np.array(aCl_Total[:, 1]),
            np.array(aCl_Total[:, 2]),
            np.array(aCl_Total[:, 3]),
        ],
        new=True,
    )
    sharp_map = hp.alm2map([almT, almE, almB], nside=nside)
    smooth = hp.smoothing(sharp_map, np.radians(0.5))

    index = np.array(
        [np.argsort(smooth[0])[-no_spots:], np.argsort(smooth[0])[:no_spots]]
    )

    # Initialize the final stacked array
    stacked = np.zeros((2, 5, 200, 200))

    def process_peak(i, j):
        lon, lat = hp.pix2ang(nside, index[j, i], lonlat=True)
        pos = hp.ang2vec(lon, lat, lonlat=True)
        neigh = hp.query_disc(nside, pos, np.radians(np.sqrt(2 * 2.5**2)))
        neigh_lon, neigh_lat = hp.pix2ang(nside, neigh, lonlat=True)
        phi = east_phi(lon, lat, neigh_lon, neigh_lat)
        empty = np.zeros((5, hp.nside2npix(nside)))
        result = np.zeros((5, 200, 200))  # Temporary array for this peak

        for k in range(5):
            if k < 3:
                empty[k, neigh] = smooth[k, neigh]
            elif k == 3:
                empty[k, neigh] = Qr(smooth[1, neigh], smooth[2, neigh], phi)
            else:
                empty[k, neigh] = Ur(smooth[1, neigh], smooth[2, neigh], phi)

            gnom_map = hp.gnomview(
                empty[k, :],
                rot=(lon, lat),
                reso=5 * 60 / 200,
                return_projected_map=True,
                no_plot=True,
            )
            result[k, :, :] = gnom_map

        return j, result

    # Set multiprocessing start method to 'spawn' or 'forkserver'
    mp.set_start_method("spawn", force=True)

    with parallel_backend("loky"):
        results = Parallel(n_jobs=-1)(
            delayed(process_peak)(i, j)
            for i in range(len(index[0, :]))
            for j in range(2)
        )

    # Combine results into the final stacked array
    for j, result in results:
        stacked[j, :, :, :] += result

    stacked /= no_spots
    return stacked


# Function to create vector map
def vectormap(step, Q, U):
    sample_row = np.arange(0, 200, step)
    sample_col = np.arange(0, 200, step)
    psi, P = pol_vec(
        Q[np.ix_(sample_row, sample_col)], U[np.ix_(sample_row, sample_col)]
    )
    x, y = np.meshgrid(
        np.arange(-2.5, 2.5, step / 200 * 5), np.arange(-2.5, 2.5, step / 200 * 5)
    )
    u = P * np.cos(psi)
    v = P * np.sin(psi)
    return x, y, u, v


average = stack_cmb_params(20000)


step = 8


# Compute vectormap data once
def compute_vectormaps():
    x_dict, y_dict, u_dict, v_dict = {}, {}, {}, {}
    for minmax in range(2):
        x, y, u, v = vectormap(step, average[minmax, 1, :, :], average[minmax, 2, :, :])
        x_dict[minmax] = x
        y_dict[minmax] = y
        u_dict["Q"] = u
        v_dict["Q"] = v
        u_dict["U"] = u
        v_dict["U"] = v

        x, y, ur, vr = vectormap(
            step, average[minmax, 3, :, :], average[minmax, 4, :, :]
        )
        u_dict["Qr"] = ur
        v_dict["Qr"] = vr
        u_dict["Ur"] = ur
        v_dict["Ur"] = vr

    return x_dict, y_dict, u_dict, v_dict


x_dict, y_dict, u_dict, v_dict = compute_vectormaps()

# Create figures
fig, ax = plt.subplots(5, 2, figsize=(16, 24), dpi=300)
fig_transformed, ax_transformed = plt.subplots(5, 2, figsize=(16, 24), dpi=300)


# Plotting function to reduce redundancy
def plot_param(ax, im_data, x, y, u, v, params, minmax, quiver_params=None):
    im = ax.imshow(im_data, extent=(-2.5, 2.5, -2.5, 2.5), cmap="jet")
    ax.grid()
    ax.set_xlabel("Degrees")
    ax.set_ylabel("Degrees")
    ax.set_title(f"Average of the {'maxima' if minmax == 0 else 'minima'} of {params}")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im, cax=cax, format="%.2f")

    if quiver_params:
        if quiver_params == "Q":
            ax.quiver(
                x, y, u_dict["Q"], v_dict["Q"], scale=15, headwidth=1, color="black"
            )
        elif quiver_params == "U":
            ax.quiver(
                x, y, u_dict["U"], v_dict["U"], scale=15, headwidth=1, color="black"
            )
        elif quiver_params == "Qr":
            ax.quiver(
                x, y, u_dict["Qr"], v_dict["Qr"], scale=15, headwidth=1, color="black"
            )
        elif quiver_params == "Ur":
            ax.quiver(
                x, y, u_dict["Ur"], v_dict["Ur"], scale=15, headwidth=1, color="black"
            )


for minmax in range(2):
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
        # Main plot
        quiver_params = None
        if params == 1:
            quiver_params = "Q"
        elif params == 2:
            quiver_params = "U"
        elif params == 3:
            quiver_params = "Qr"
        elif params == 4:
            quiver_params = "Ur"

        plot_param(
            ax[params, minmax],
            average[minmax, params, :, :],
            x_dict[minmax],
            y_dict[minmax],
            u_dict.get(name, None),
            v_dict.get(name, None),
            name,
            minmax,
            quiver_params,
        )

        # Transformed-only plot
        quiver_params = None
        if params == 3:
            quiver_params = "Qr"
        elif params == 4:
            quiver_params = "Ur"

        plot_param(
            ax_transformed[params, minmax],
            average[minmax, params, :, :],
            x_dict[minmax],
            y_dict[minmax],
            u_dict.get(name, None),
            v_dict.get(name, None),
            name,
            minmax,
            quiver_params,
        )

# Save figures
fig.savefig("Output/Average.png")
fig_transformed.savefig("Output/transformed_only.png")

plt.show()
