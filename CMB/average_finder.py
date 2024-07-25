# Importing modules required for stacking
import numpy as np
import healpy as hp
import camb
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from numba import jit
import time

# Importing modules required for maps
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Functions needed for stacking


# Define Qr and Ur parameters using Numba
@jit(nopython=True, parallel=True)
def Qr(Q, U, phi):
    return -Q * np.cos(2 * phi) - U * np.sin(2 * phi)


@jit(nopython=True, parallel=True)
def Ur(Q, U, phi):
    return Q * np.sin(2 * phi) - U * np.cos(2 * phi)


# Generate polarization vectors using Numba
@jit(nopython=True)
def pol_vec(Q, U):
    phi = 0.5 * np.arctan2(U, Q)
    P = np.sqrt(Q**2 + U**2)
    return phi, P


# Function to create inputs for vector map
def vectormap(step, Q, U):
    sample_row = np.arange(0, 200, step)
    sample_col = np.arange(0, 200, step)
    phi, P = pol_vec(
        Q[np.ix_(sample_row, sample_col)],
        U[np.ix_(sample_row, sample_col)],
    )
    x, y = np.meshgrid(
        np.arange(-2.5, 2.5, step / 200 * 5), np.arange(-2.5, 2.5, step / 200 * 5)
    )
    u = -P * np.cos(2 * phi)
    v = P * np.sin(2 * phi)
    return x, y, u, v


# Compute vectormap data once
def compute_vectormaps(average, step):
    x_dict, y_dict, u_dict, v_dict = {}, {}, {}, {}
    for minmax in range(2):
        x, y, ur, vr = vectormap(
            step,
            average[minmax, 1, :, :],
            average[minmax, 2, :, :],
        )
        x_dict[minmax] = x
        y_dict[minmax] = y
        u_dict["Q"] = ur
        v_dict["Q"] = vr
        u_dict["U"] = ur
        v_dict["U"] = vr

    return x_dict, y_dict, u_dict, v_dict


# Function for plotting
def plot_param(ax, im_data, x, y, u, v, params, minmax, quiver_params=None):
    im = ax.imshow(
        im_data, extent=(-2.5, 2.5, -2.5, 2.5), origin="lower", cmap="coolwarm"
    )
    ax.grid()
    ax.set_xlabel("Degrees")
    ax.set_ylabel("Degrees")
    ax.set_title(f"Average of the {'maxima' if minmax == 0 else 'minima'} of {params}")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="20%", pad=0.05)
    plt.colorbar(im, cax=cax, format="%.2f")

    if quiver_params:
        ax.quiver(x, y, u, v, scale=10, headwidth=1, color="black")


# Function to compute data for one peak with an index
def process_peak(smooth_map, index, minmax, j, nside):
    lon, lat = hp.pix2ang(nside, index[minmax, j], lonlat=True)
    rot = hp.Rotator(rot=[lon, lat], deg=True)
    cmap = rot.rotate_map_alms(smooth_map)
    neigh = hp.query_disc(512, np.array([1, 0, 0]), np.radians(np.sqrt(2 * 2.5**2)))
    neigh_lon, neigh_lat = hp.pix2ang(nside, neigh, lonlat=True)
    phi = np.arctan2(neigh_lat, np.where(neigh_lon < 180, neigh_lon, neigh_lon - 360))
    empty_original = np.zeros((3, hp.nside2npix(nside)))
    pol_map = np.zeros((2, hp.nside2npix(nside)))
    # made map of only the neighbours of peak in original coordinates
    for pindx in range(3):
        empty_original[pindx, neigh] = smooth_map[pindx, neigh]
    pol_map[0, neigh] = Qr(cmap[1, neigh], cmap[2, neigh], phi)
    pol_map[1, neigh] = Ur(cmap[1, neigh], cmap[2, neigh], phi)
    result = np.zeros((5, 200, 200))
    for pindx in range(5):
        if pindx < 3:
            gnom_map = hp.gnomview(
                cmap[pindx, :],
                reso=5 * 60 / 200,
                return_projected_map=True,
                no_plot=True,
            )
        elif pindx == 3:
            gnom_map = hp.gnomview(
                pol_map[0, :],
                reso=5 * 60 / 200,
                return_projected_map=True,
                no_plot=True,
            )
        else:
            gnom_map = hp.gnomview(
                pol_map[1, :],
                reso=5 * 60 / 200,
                return_projected_map=True,
                no_plot=True,
            )
        result[pindx, :, :] = gnom_map
    return minmax, result


def stack_cmb_params(no_spots, lensing=True, nside=512):
    planck2018pars = camb.read_ini("planck_2018.ini")
    planck2018 = camb.get_results(planck2018pars)
    powers = planck2018.get_cmb_power_spectra(
        planck2018pars, CMB_unit="muK", raw_cl=True
    )
    if lensing:
        aCl_Total = powers["total"]
    else:
        aCl_Total = powers["unlensed_total"]
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

    stacked = np.zeros((2, 5, 200, 200))

    mp.set_start_method("spawn", force=True)
    with parallel_backend("loky"):
        results = Parallel(n_jobs=10, timeout=600)(
            delayed(process_peak)(smooth, index, minmax, j, nside)
            for minmax in range(2)
            for j in range(no_spots)
        )
    for minmax, result in results:
        stacked[minmax, :, :, :] += result
    stacked /= no_spots
    return stacked


# Run function
peaks = 500

start_time = time.time()
lensed = stack_cmb_params(peaks, lensing=True)
end_time = time.time()
print(f"Runtime for lensed stack: {end_time - start_time} seconds")

# start_time = time.time()
# nolens = stack_cmb_params(peaks, lensing=False)
# end_time = time.time()
# print(f"Runtime for nolens stack: {end_time - start_time} seconds")

step = 8
x_dict, y_dict, ul_dict, vl_dict = compute_vectormaps(lensed, step)
# x_dict, y_dict, un_dict, vn_dict = compute_vectormaps(nolens, step)

figl, axl = plt.subplots(5, 2, figsize=(16, 24), dpi=300)
# fign, axn = plt.subplots(5, 2, figsize=(16, 24), dpi=300)

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
        quiver_params = None
        if params == 3:
            quiver_params = "Q"
        elif params == 4:
            quiver_params = "U"

        plot_param(
            axl[params, minmax],
            lensed[minmax, params, :, :],
            x_dict[minmax],
            y_dict[minmax],
            ul_dict.get(quiver_params, None),
            vl_dict.get(quiver_params, None),
            name,
            minmax,
            quiver_params,
        )

""""
        plot_param(
            axn[params, minmax],
            nolens[minmax, params, :, :],
            x_dict[minmax],
            y_dict[minmax],
            un_dict.get(quiver_params, None),
            vn_dict.get(quiver_params, None),
            name,
            minmax,
            quiver_params,
        )
        """

figl.savefig("Output/Lensed_Average.png")
# fign.savefig("Output/Nolensed_Average.png")

plt.show()
