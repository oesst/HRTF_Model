import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import src.features.helpers_vis as hp_vis
import src.features.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click

hp_vis.set_layout(15)

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))


def plot_corrcoeff(map, ax):

    c = ax.pcolormesh(map, vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(c)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('Correlation Coefficient',  labelpad=10, rotation=270)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_xticklabels(['', 'HRTF_C', 'HRTF_I', 'Learned MAP', ''])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.set_yticklabels(['', 'HRTF_C', 'HRTF_I', 'Learned MAP', ''])

    return ax


# Define whether figures should be saved
@click.command()
@click.option('--save_figs', type=click.BOOL, default=False, help='Save figures')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
@click.option('--model_name', default='single_participant', help='Defines the model name.')
@click.option('--exp_name', default='single_participant_default', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--participant_number', default=9, help='CIPIC participant number. Default is 9')
@click.option('--snr', default=0.2, help='Signal to noise ration to use. Default is 0.2')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--sigma_smoothing', default=0, help='Sigma for smoothing kernel. 0 is off. Default is 0.')
@click.option('--sigma_gauss_norm', default=1, help='Sigma for gauss normalization. 0 is off. Default is 1.')
def main(save_figs=False, save_type='svg', model_name='hrtf_comparison', exp_name='single_participant', azimuth=12, participant_number=9, snr=0.2, freq_bands=24, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1):
    logger = logging.getLogger(__name__)
    logger.info('Showing Correlation Coefficient Maps between HRTFs and differntly learned Maps')

    # make sure save type is given
    if not save_type or len(save_type) == 0:
        save_type = 'svg'

    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, elevations, 1)

    ######################################

    # create unique experiment name
    exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map, time_window, int(
        snr * 100), freq_bands, max_freq, participant_number, (azimuth - 12) * 10, normalize, len(elevations)])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [hrtfs_i, hrtfs_c, learned_map_mono, learned_map_mono_mean, learned_map_bin, learned_map_bin_mean] = pickle.load(f)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('Mono Map')
        tmp = np.concatenate((hrtfs_c, hrtfs_i, learned_map_mono))
        tmp = np.corrcoef(tmp)
        plot_corrcoeff(tmp, ax)

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('Mono-Mean Map')
        tmp = np.concatenate((hrtfs_c, hrtfs_i, learned_map_mono_mean))
        tmp = np.corrcoef(tmp)
        plot_corrcoeff(tmp, ax)

        ax = fig.add_subplot(2, 2, 3)
        ax.set_title('Bin Map')
        tmp = np.concatenate((hrtfs_c, hrtfs_i, learned_map_bin))
        tmp = np.corrcoef(tmp)
        plot_corrcoeff(tmp, ax)

        ax = fig.add_subplot(2, 2, 4)
        ax.set_title('Bin-Mean Map')
        tmp = np.concatenate((hrtfs_c, hrtfs_i, learned_map_bin_mean))
        tmp = np.corrcoef(tmp)
        plot_corrcoeff(tmp, ax)

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name / exp_name_str
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            logger.info('Saving figures to ' + fig_save_path.as_posix())
            plt.savefig((fig_save_path / (exp_name + '_hrtfs_xcorr.' + save_type)).as_posix(), dpi=300)

        plt.show()
    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
