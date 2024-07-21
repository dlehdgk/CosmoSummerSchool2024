import numpy as np
import camb
import healpy as hp
from camb import model, initialpower
from numba import jit


# Define Qr and Ur parameters
@jit(nopython=True)
def Qr(Q, U, phi):
    Qr = -Q * np.cos(2 * phi) - U * np.sin(2 * phi)
    return Qr


@jit(nopython=True)
def Ur(Q, U, phi):
    Ur = Q * np.sin(2 * phi) - U * np.cos(2 * phi)
    return Ur


# function which takes hp.pix2ang(lonlat=True) outputs of longitude and latitude in degrees and converts them into an angle from the east from the central point
# angle is in radians
@jit(nopython=True)
def east_phi(lon_c, lat_c, lon_p, lat_p):
    dlon = np.radians(lon_p - lon_c)
    dlat = np.radians(lat_p - lat_c)
    # angle from east with east pointing left
    return np.arctan2(dlat, -dlon)


# create function which generates polarisation vectors
@jit(nopython=True)
def pol_vec(Q, U):
    psi = 0.5 * np.arctan2(U, Q)
    P = np.sqrt(Q**2 + U**2)
    return psi, P


def stack_cmb_params(no_spots, param_file="planck_2018.ini", nside=512):
    # Use input initial params to generate cmb spectra
    planck2018pars = camb.read_ini(param_file)
    planck2018 = camb.get_results(planck2018pars)
    # use unnormalised Cl
    powers = planck2018.get_cmb_power_spectra(
        planck2018pars, CMB_unit="muK", raw_cl=True
    )
    # total power spectrum
    aCl_Total = powers["total"]
    # l starts from 0 (monopole)
    lmax = aCl_Total.shape[0] - 1
    # l steps
    aL = np.arange(lmax + 1)
    # generate alm for T E B using power spectra
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

    # finding no_spots peaks from top and bottom temperature values
    # indices of the spots 0 = top, 1 = bottom
    index = np.array(
        [np.argsort(smooth[0])[-no_spots:], np.argsort(smooth[0])[:no_spots]]
    )

    # generate array for final averaged map data
    # 0 = max 1 = min
    # 0 = temperature, 1 = Q, 2=U, 3=Qr, 4=Ur
    stacked = np.zeros((2, 5, 200, 200))

    # for loop to identify and sum all peaks and neighbours
    for i in range(len(index[0, :])):
        # sum for max and min
        for j in range(2):
            # finding angular positions of pixels in longitude and latitude degrees
            lon, lat = hp.pix2ang(nside, index[j, i], lonlat=True)
            # find position vector of given angular positions
            pos = hp.ang2vec(lon, lat, lonlat=True)
            # finding neighbours in a 5x5 degree area centred at the central point
            neigh = hp.query_disc(nside, pos, np.radians(np.sqrt(2 * 2.5**2)))
            # position of all neighbour points
            neigh_lon, neigh_lat = hp.pix2ang(nside, neigh, lonlat=True)
            # creating an empty map to insert local values
            empty = np.zeros((5, hp.nside2npix(nside)))
            # angle from the east
            phi = east_phi(lon, lat, neigh_lon, neigh_lat)
            # for loop for each parameter
            for k in range(5):
                if k < 3:
                    empty[k, neigh] = smooth[k, neigh]
                elif k == 3:
                    empty[k, neigh] = Qr(smooth[1, neigh], smooth[2, neigh], phi)
                else:
                    empty[k, neigh] = Ur(smooth[1, neigh], smooth[2, neigh], phi)
                # forming a gnomomic map centred at the peak
                stacked[j, k, :, :] += hp.gnomview(
                    empty[k, :],
                    rot=(lon, lat),
                    reso=5 * 60 / 200,
                    return_projected_map=True,
                    no_plot=True,
                )
    stacked /= no_spots
    # returns array of averaged points around the peak of dimensions [min/max, params, x, y]
    return stacked


# Function which takes staked map of Q and U and returns input for plt.quiver


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
