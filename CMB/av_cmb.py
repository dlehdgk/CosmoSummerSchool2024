import multiprocessing as mp

import camb
import healpy as hp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import jit


class cmbav:
    def __init__(self,cmb_ini,cpu_cores, peaks, size, lensing=True, nside=512):
        pars = camb.read_ini(cmb_ini)
        data = camb.get_results(pars)
        power = data.get_cmb_power_spectra(self.__pars, CMB_unit="muK", raw_cl=True)
        if lensing:
            self.aCl = power["Total"]
        else:
            self.aCl = power["unlensed_total"]
        self.powers = 
        self.__cpu = cpu_cores
        self.__peaks = peaks
        self.__size = size
        self.__nside = nside

    # function to process one peak
    def process_peak(smooth_map, index, minmax, j, nside):
    lon, lat = hp.pix2ang(nside, index[minmax, j], lonlat=True)
    # rotate data to be centred on the peak
    rot = hp.Rotator(rot=[lon, lat], deg=True)
    cmap = rot.rotate_map_alms(smooth_map)
    # find the neighbours around the peak (vector = (1,0,0))
    neigh = hp.query_disc(nside, np.array([1, 0, 0]), np.radians(np.sqrt(2 * (self.__size/2)**2)))
    neigh_lon, neigh_lat = hp.pix2ang(nside, neigh, lonlat=True)
    # compute phi noting longitude goes from 0 to 360
    phi = np.arctan2(neigh_lat, np.where(neigh_lon < 180, neigh_lon, neigh_lon - 360))

    pol_map = np.zeros((2, hp.nside2npix(nside)))
    pol_map[0, neigh] = Qr(cmap[1, neigh], cmap[2, neigh], phi)
    pol_map[1, neigh] = Ur(cmap[1, neigh], cmap[2, neigh], phi)
    # made map of only the neighbours of peak in original coordinates
    result = np.zeros((5, 200, 200))
    for pindx in range(5):
        if pindx < 3:
            gnom_map = hp.gnomview(
                cmap[pindx, :],
                reso=5 * 60 / 200,
                return_projected_map=True,
                no_plot=True,
            )
        else:
            gnom_map = hp.gnomview(
                pol_map[pindx - 3, :],
                reso=5 * 60 / 200,
                return_projected_map=True,
                no_plot=True,
            )
        result[pindx, :, :] = gnom_map
    return minmax, result
    
    def stack_cmb_params(self):
        almT,almE,almB = hp.synalm([np.array(self.aCl[:,0]),np.array(self.aCl[:,1]),np.array(self.aCl[:,2]),np.array(self.aCl[:,3])],new=True)
        sharp_map = hp.alm2map([almT,almE,almB],nside=self.__nside)
        # smoothed map to 1/10th of measured scale
        smooth_map = hp.smoothing(sharp_map,np.radians(0.1*self.__size))
        # finding the indices of the temperature peaks
        index = np.array([np.argsort(smooth_map[0])[-self.__peaks:], np.argsort(smooth_map[0])[:self.__peaks]])





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
        results = Parallel(n_jobs=cpu, timeout=600)(
            delayed(process_peak)(smooth, index, minmax, j, nside)
            for minmax in range(2)
            for j in range(no_spots)
        )
    for minmax, result in results:
        stacked[minmax, :, :, :] += result
    stacked /= no_spots
    return stacked


# Run function
peaks = 20000

start_time = time.time()
lensed = stack_cmb_params(peaks, lensing=True)
end_time = time.time()
print(f"Runtime for lensed stack: {end_time - start_time} seconds")

# saving data for analysis as csv with a file for array shape
df_lensed = pd.DataFrame(lensed.reshape(-1, lensed.shape[-1]))
df_lensed.to_csv("Output/lensed.csv", index=False)
lensed_shape = "lensed_shape.txt"
with open(lensed_shape, "w") as f:
    f.write(",".join(map(str, lensed.shape)))

start_time = time.time()
nolens = stack_cmb_params(peaks, lensing=False)
end_time = time.time()
print(f"Runtime for nolens stack: {end_time - start_time} seconds")

# saving data for analysis as csv with a file for array shape
df_nolens = pd.DataFrame(nolens.reshape(-1, nolens.shape[-1]))
df_nolens.to_csv("Output/nolens.csv", index=False)
nolens_shape = "nolens_shape.txt"
with open(nolens_shape, "w") as f:
    f.write(",".join(map(str, nolens.shape)))

step = 8
x_dict, y_dict, ul_dict, vl_dict = compute_vectormaps(lensed, step)
x_dict, y_dict, un_dict, vn_dict = compute_vectormaps(nolens, step)

figl, axl = plt.subplots(5, 2, figsize=(16, 24), dpi=300)
fign, axn = plt.subplots(5, 2, figsize=(16, 24), dpi=300)

for minmax in range(2):
    for params, name in enumerate(
        [
            "Temperature",
            "Q Polarisation",
            "U Polarisation",
            "Qr Polarisation",
            "Ur Polarisation",
        ]
    ):
        quiver_required = params in [3, 4]

        plot_param(
            axn[params, minmax],
            nolens[minmax, params, :, :],
            x_dict[minmax],
            y_dict[minmax],
            un_dict[minmax] if quiver_required else None,
            vn_dict[minmax] if quiver_required else None,
            name,
            minmax,
            quiver_required,
        )

        plot_param(
            axl[params, minmax],
            lensed[minmax, params, :, :],
            x_dict[minmax],
            y_dict[minmax],
            ul_dict[minmax] if quiver_required else None,
            vl_dict[minmax] if quiver_required else None,
            name,
            minmax,
            quiver_required,
        )
