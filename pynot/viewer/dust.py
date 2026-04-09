
import numpy as np
from scipy.interpolate import UnivariateSpline as spline

parameters = {
    'SMC': {'c1': -4.959, 'c2': 2.264, 'c3': 0.389, 'c4': 0.0, 'x0': 4.579, 'gamma': 0.934, 'Rv': 2.74},
    'LMC': {'c1': -0.89, 'c2': 0.998, 'c3': 2.719, 'c4': 0.0, 'x0': 4.579, 'gamma': 0.934, 'Rv': 3.41},
    }


def MW_reddening(wl, Rv=3.1):
    """
    Average Galactic reddening law parametrized by Fitzpatrick & Massa (2007).

    INPUT
    wl: wavelength in Angstrom

    Rv: if not given, the average value of 3.1 is assumed.

    Returns A(l)/A(V) evaluated at the input wavelengths.
    """

    if isinstance(wl, float):
        wl = np.array([wl])
        convert2float = True
    else:
        convert2float = False

    k_in = 1./(wl*1.e-4)

    c4 = 0.319
    c3 = 2.991
    c2 = 5.0/Rv - 0.85
    c1 = 2.09 - 2.84*c2
    x0 = 4.592
    gamma = 0.922
    pars_avg = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
                'x0': x0, 'gamma': gamma, 'Rv': Rv}

    ksi = FM2007(k_in, **pars_avg)

    if convert2float:
        ksi = float(ksi)
    return ksi


def FM2007(wl, c1, c2, c3, c4, Rv, x0=4.579, gamma=0.934):
    """
    Fitzpatrick & Mazza extinction curve
    Return ksi = A(lambda)/A(V) instead of E(B-lambda)/E(B-V).
    Using O2 and O3 as free parameters for optical spline.
      O2 = 1.33 +/- 0.01
      O3 = 2.0 +/- 0.1
    O1 is kept fixed to ensure correct normalization.

    Parameters
    ----------
    wl : np.ndarray
        Wavelength array in angstrom

    pars : dict
        Parameters for the extinction curve:
        (c1, c2, c3, c4, x0, gamma, Rv)

    Returns
    -------
    ksi : np.ndarray
        Array of A(lambda) / A(v) for the given input wavelength array.
    """
    k = 1 / (wl*1.e-4)

    D = k**2/((k**2 - x0**2)**2 + k**2*gamma**2)

    F = 0.5392*(k-5.9)**2 + 0.05644*(k-5.9)**3
    F[k < 5.9] = 0.

    # Use Fitzpatrick & Massa 1990 original formulation
    # with fixed UV polynomial:
    Ebv = c1 + c2*k + c3*D + c4*F

    # Use IR power-law from Fitzpatrick & Massa 2007
    # assuming the correlation ketween k_IR and Rv.
    # Their eq. 7:
    Ebv_IR = (-0.83 + 0.63*Rv)*k**1.84 - Rv

    # Use spline points from 1 < x < 3.7
    # Anchor to UV and IR parts to make smooth transition: (FM2007)
    U1 = 3.85                                    # anchor at 2600A
    U2 = 3.7                                    # anchor at 2700A
    O_UV1 = c1 + c2*U1 + c3*U1**2/((U1**2 - x0**2)**2 + U1**2*gamma**2)
    O_UV2 = c1 + c2*U2 + c3*U2**2/((U2**2 - x0**2)**2 + U2**2*gamma**2)
    O_IRopt = (-0.83 + 0.63*Rv)*1.0**1.84 - Rv    # anchor at 1.0
    O_IR = (-0.83 + 0.63*Rv)*0.75**1.84 - Rv    # anchor at 0.75

    # Array of anchor points for spline
    O = np.array([O_IR, O_IRopt, 0., O_UV2, O_UV1])

    # Array of inverse wavelength for spline anchors:
    k_anchor = np.array([0.75, 1., 1.808, U2, U1])

    Ebv_spline_func = spline(k_anchor, O)
    Ebv_spline = Ebv_spline_func(k)

    # stitch together the pieces
    Ebv = Ebv*(k >= 3.7) + Ebv_spline*(k < 3.7)*(k > 1.) + Ebv_IR*(k <= 1.)

    ksi = Ebv/Rv + 1.

    return ksi


class DustModel:
    name = ''

    def __init__(self, name: str, pars: dict[str, float], functor=FM2007):
        self.name = name
        self.pars = pars
        self.functor = functor

    def __call__(self, x: np.ndarray, Av=1.0):
        model = 10**(-0.4 * self.functor(x, **self.pars) * Av)
        return 10**(-0.4 * self.functor(x, **self.pars) * Av)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.name


class SMCDustModel(DustModel):
    name = 'SMC'

    def __init__(self):
        super().__init__(name='SMC', pars=parameters['SMC'])


class LMCDustModel(DustModel):
    name = 'LMC'

    def __init__(self):
        super().__init__(name='LMC', pars=parameters['LMC'])


class MWDustModel(DustModel):
    name = 'MW'

    def __init__(self):
        super().__init__(name='MW', pars={'Rv': 3.1}, functor=MW_reddening)
