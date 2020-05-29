import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import src.visualization.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click

hp.set_layout(15)

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))


# Define whether figures should be saved
@click.command()
@click.option('--save_figs', default=False, help='Save the figures.')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
def main(save_figs=False, save_type='svg', model_name='hrtf_comparison', exp_name='single_participant'):

    logger = logging.getLogger(__name__)
    logger.info('Showing Correlation Coefficient Maps between HRTFs and differntly learned Maps')

    ### Set Parameters of Input Files ###
    azimuth = 12
    snr = 0
    freq_bands = 128
    participant_number = 9
    max_freq  = 20000
    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, 25, 1)

    # filtering parameters
    normalization_type = 'sum_1'
    sigma_smoothing = 0
    sigma_gauss_norm = 1

    # use the mean subtracted map as the learned map
    mean_subtracted_map = True

    ear = 'ipsi'
    ######################################

    # create unique experiment name
    exp_name_str = exp_name + '_' + normalization_type + str(sigma_smoothing) + str(sigma_gauss_norm) + str(mean_subtracted_map) + '_' + str(time_window) + '_window_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_'+str(max_freq)+'_max_freq_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm' + str(len(elevations)) + '_elevs.npy'

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [hrtfs_i, hrtfs_c, learned_map_mono, learned_map_mono_mean, learned_map_bin, learned_map_bin_mean] = pickle.load(f)

        fig = plt.figure(figsize=(20, 5))
        plt.suptitle('Participant: ' + str(participant_number))

        # Monoaural Data (Ipsilateral), No Mean Subtracted
        ax = fig.add_subplot(1, 2, 1)
        # a = ax.pcolormesh(np.squeeze(hrtfs_i[:,:-5]))
        data = hrtfs_i[:, 5:-5]
        a = ax.pcolormesh(np.linspace(0, 1, data.shape[1]), np.linspace(-45, 90, data.shape[0]),
                          data, shading='gouraud', linewidth=0, rasterized=True)
        formatter = hp.ERBFormatter(100, 18000, unit='', places=0)
        ax.xaxis.set_major_formatter(formatter)
        plt.colorbar(a)
        ax.set_title('Ipsilateral')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Elevations [deg]')

        ax = fig.add_subplot(1, 2, 2)
        data = hrtfs_c[:, 5:-5]
        a = ax.pcolormesh(np.linspace(0, 1, data.shape[1]), np.linspace(-45, 90, data.shape[0]),
                          data, shading='gouraud', linewidth=0, rasterized=True)
        formatter = hp.ERBFormatter(100, 18000, unit='', places=0)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_title('Contralateral')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Elevations [deg]')

        plt.colorbar(a)

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name / exp_name_str
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            logger.info('Saving figures to ' + fig_save_path.as_posix())
            plt.savefig((fig_save_path / (exp_name +'_'+str(participant_number)+ '_raw_hrtfs.' + save_type)).as_posix(), dpi=300, transparent=True)
            plt.close()

        plt.show()

    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
