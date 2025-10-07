from contourpy import contour_generator
import numpy as np
from scipy import interpolate as sipt


def mdot0(gamma):
    return (2 / (5 - 3 * gamma)) ** ((5 - 3 * gamma) / (2 * gamma - 2)) / 4.0


def B0(x, y, gamma, mdot):
    return (
        y / 2.0 - 1.0 / x + (mdot / (x * x * np.sqrt(y))) ** (gamma - 1) / (gamma - 1)
    )


def B0_inf(gamma):
    return 1 / (gamma - 1)


def ys(gamma, mdot):
    return (4 * mdot) ** ((2 * gamma - 2) / (5 - 3 * gamma))


def xs(gamma, mdot):
    return 1 / (2 * ys(gamma, mdot))


def calculate_flow(gamma, mdot, range):
    xvals: np.ndarray
    yvals: np.ndarray
    rs = xs(gamma, mdot)
    if range == "sup":
        xvals = np.logspace(-3, np.log10(0.95 * rs), 500)
        yvals = np.logspace(0, 5, 1000)
    elif range == "sub":
        xvals = np.logspace(np.log10(1.1 * rs), 2, 500)
        yvals = np.logspace(-10, 3, 1000)
    else:
        raise Exception("Range must be chosen. That is either subsonic or supersonic")
    X, Y = np.meshgrid(xvals, yvals)
    Z = np.array([[B0(x, y, gamma, mdot) for x in xvals] for y in yvals])

    cont_gen = contour_generator(z=Z, x=X, y=Y)
    contour_lines = cont_gen.lines(B0_inf(gamma))
    if len(contour_lines) >= 2:
        assert type(contour_lines) is list
        if np.size(contour_lines[0]) > np.size(contour_lines[1]):
            return contour_lines[0]
        else:
            return contour_lines[1]
    else:
        return contour_lines[0]


def extract(arr):
    t1 = arr[:, 0]
    t2 = arr[:, 1]
    mask1 = np.logical_and(np.diff(t2) > 0, np.diff(t1) < 0)
    mask2 = np.logical_and(np.diff(t2) < 0, np.diff(t1) > 0)
    totalmask = mask1 | mask2
    indices = np.where(totalmask)
    t1 = t1[indices]
    t2 = t2[indices]
    return t1, t2


def soln(gamma, arr=None):
    mdotval = mdot0(gamma)
    c1 = calculate_flow(gamma, mdot0(gamma), "sub")
    x1, y1 = extract(c1)
    c2 = calculate_flow(gamma, mdot0(gamma), "sup")
    x2, y2 = extract(c2)
    ind1 = np.argsort(x1)
    ind2 = np.argsort(x2)
    x1 = x1[ind1]
    x2 = x2[ind2]
    y1 = y1[ind1]
    y2 = y2[ind2]
    x = np.concatenate((x2, x1))
    y = np.concatenate((y2, y1))
    y = np.sqrt(y)
    if arr is None:
        return x, y
    else:
        bip = sipt.interp1d(np.log10(x), np.log10(y))
        res = 10.0 ** (bip(np.log10(arr)))
        u_res = np.ascontiguousarray(res)
        rho_res = mdotval / (arr**2 * u_res)

        return mdotval, u_res, rho_res
