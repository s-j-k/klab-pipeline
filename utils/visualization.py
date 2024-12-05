import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec
from utils.curate import during
from scipy.ndimage import gaussian_filter1d


class ShapeError(Exception):
    pass


def correct_img(mat):
    mat_ = mat.copy()
    mat_ = mat_.clip(min=np.percentile(mat_, 1), max=np.percentile(mat_, 99))
    mat_ = (mat_ - mat_.min()) / (mat_.max() - mat_.min())
    mat_ = (mat_-0.5) * 0.8 + 0.5
    return mat_


def _fn_plot_tone_evoked_traces(ax, s, dff):
    samples_per_color = 80
    evoked = []
    for tone in np.unique(s):
        evoked.append(dff[s == tone, 20:(20+samples_per_color)])
    evoked = np.array(evoked)

    evoked = np.transpose(evoked, (1, 0, 2))
    evoked = np.reshape(evoked, [evoked.shape[0], -1])
    ax.plot(evoked.T, c='k', alpha=0.1)

    cmap = plt.get_cmap('jet')
    num_colors = 17
    colors = cmap(np.linspace(0, 1, num_colors))
    for i in range(0, evoked.shape[1], samples_per_color):
        end_idx = min(i + samples_per_color, evoked.shape[1])
        ax.plot(range(i, end_idx), evoked.mean(0)[i:end_idx], color=colors[i // samples_per_color], lw=1)
    for m in np.arange(samples_per_color, num_colors * samples_per_color, samples_per_color):
        ax.axvline(m, ls=':', c='k', alpha=0.3)
    ax.axhline(0, c='k', alpha=0.3)
    ax.set_xlim(0, evoked.shape[1]-1)
    ax.set_xticks(np.arange(samples_per_color/2, (2 * num_colors + 1) * samples_per_color / 2, samples_per_color))
    ax.set_xticklabels([f"{x/1000:0.01f}KHz" for x in np.unique(s)], rotation=45)


def _fn_get_roi_overlay(peak_freq, roi_masks, sz):
    freq = np.array(peak_freq)
    freq[freq < np.log10(4000)] = np.nan
    freq[freq > np.log10(64000)] = np.nan

    mask = np.nan * np.zeros(sz)
    for i_m, m in enumerate(roi_masks):
        for triplet in m:
            mask[triplet[0], triplet[1]] = freq[i_m]
    return mask


def _fn_overlay_mask_over_img(ax, bg, masks, mask_labels, cmap='gray', clim=(4000, 64000), alpha=0.5):
    ax.imshow(bg, cmap=cmap, alpha=alpha)

    im = []
    for mask, mask_label in zip(masks, mask_labels):
        overlay = _fn_get_roi_overlay(mask_label, mask, bg.shape[:2])
        im.append(ax.imshow(overlay, cmap='jet', vmin=np.log10(clim[0]), vmax=np.log10(clim[1])))
    ax.grid(False)
    ax.axis('off')
    return im


def _fn_plot_highlight_roi(ax, bg, m):
    overlay_bg = _fn_overlay_roi_over_bg(m, bg)
    ax.imshow(overlay_bg, cmap='gray')
    ax.axis('off')


def _fn_plot_tuning_curve(ax, d, t, s, activation_window=(0, 700)):
    df = pd.DataFrame({
        'response': d[:, during(t, *activation_window)].mean(1),
        'tone': s
    })
    sns.pointplot(data=df, x='tone', y='response', ax=ax, errorbar='se')
    ax.set_xticks([])


def _fn_overlay_roi_over_bg(pm, bg, val=0):
    bg_ = correct_img(bg) * 0.5
    pm_ = np.vstack([[x['x'], x['y'], x['weight']] for x in pm])
    bg_[pm_[:, 0].astype(int), pm_[:, 1].astype(int)] = 1
    return bg_


def _fn_plot_average_evoked_response(ax, t, d):
    avg = d.mean(0)
    sem = d.std(0) / np.sqrt(d.shape[0])

    ax.plot(t, avg)
    ax.fill_between(t, avg + sem, avg - sem, alpha=0.2)
    ax.plot(t, gaussian_filter1d(avg, sigma=3))

    ax.axhline(0, alpha=0.2, c='k')
    ax.axvline(0, alpha=0.2, c='k', ls='--')
    ax.axvline(-800, alpha=0.2, c='k', ls='--')
    ax.axvline(200, alpha=0.2, c='r', ls='--')
    ax.axvline(1000, alpha=0.2, c='r', ls='--')


def summplot(data, stim_filter=...):
    d = data['data'][stim_filter, :]
    t = data['time_range']
    s = data['event_labels'][stim_filter]
    m = data['mask']
    bg = data['background']

    gs = GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])

    _fn_plot_highlight_roi(ax1, bg, m)
    _fn_plot_average_evoked_response(ax2, t, d)
    _fn_plot_tuning_curve(ax3, d, t, s, [200, 1000])
    _fn_plot_tone_evoked_traces(ax4, s, d)

    return fig


def tonoplot(red, grn, c_mask, c_freq, a_mask, a_freq, style: str = 'merged'):
    bg = np.zeros(grn.shape + (3,))
    bg[..., 0] = correct_img(red)
    bg[..., 1] = correct_img(grn)
    bg_bw = np.array(Image.fromarray((bg * 255).astype(np.uint8)).convert('L'))

    if style == 'merged':
        fig, axes = plt.subplots(1, 1, figsize=(8, 5))

        im = _fn_overlay_mask_over_img(axes, bg_bw, [c_mask, a_mask], [c_freq, a_freq])
        cbar = fig.colorbar(im[1], ax=axes, aspect=20, shrink=0.9)
        axes.set_title('Red + Green Channels')

    elif style == 'side-by-side':
        fig, axes = plt.subplots(2, 1, figsize=[12, 16], constrained_layout=True)

        __ = _fn_overlay_mask_over_img(axes[0], bg[..., 0], [c_mask], [c_freq])
        im = _fn_overlay_mask_over_img(axes[1], bg[..., 1], [a_mask], [a_freq])
        cbar = fig.colorbar(im[0], ax=axes, fraction=0.05, shrink=0.53, pad=0.01)
        axes[0].set_title('Red Channel (AuC Cell Bodies)')
        axes[1].set_title('Green Channel (MGB Axons)')

    else:
        raise ValueError("Unknown style; supported styles are: <merged>, <side-by-side>")

    return fig, cbar