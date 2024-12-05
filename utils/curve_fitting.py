import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def gauss(x, a, mu, s, c):
    return (a ** 2) * np.exp(-(x - mu) ** 2 / (s ** 2)) + c


def fn_fit_gaussian_tuning(r, s):
    initial_guess = [0.05, 4.2, 0.1, 0]  # initial guess for parameters: a, mu, sigma, c

    u_df = pd.DataFrame({'resp': r[:, 14:22].mean(1), 'stim': s})
    m_df = u_df.groupby('stim').mean().reset_index()

    f = m_df['stim']
    log_f = np.log10(f)

    try:
        # noinspection PyTupleAssignmentBalance
        parameters, _ = curve_fit(gauss, log_f, m_df.resp, p0=initial_guess)
    except Exception:
        parameters = [np.nan, np.nan, np.nan, np.nan]
        pass

    return parameters


def fit_gaussian_tuning(epochs, stimuli):
    peak_freq, fit_results = [], []
    for i_roi in np.arange(epochs.shape[2]):
        parameters = fn_fit_gaussian_tuning(epochs[..., i_roi], stimuli)
        peak_freq.append(parameters[1])
        fit_results.append(parameters)

    return peak_freq, fit_results
