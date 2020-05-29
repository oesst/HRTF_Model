import matplotlib.pyplot as plt
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
def main(save_figs=False, save_type='svg', model_name='all_participants', exp_name='localization_default'):
    """ This script plots the localization quality for all participants over sounds
    """
    logger = logging.getLogger(__name__)
    logger.info('Showing localization results for all sounds')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    azimuth = 12
    snr = 0.2
    freq_bands = 128
    max_freq = 20000
    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
                                    12, 15, 17, 18, 19, 20,
                                    21, 27, 28, 33, 40, 44,
                                    48, 50, 51, 58, 59, 60,
                                    61, 65, 119, 124, 126,
                                    127, 131, 133, 134, 135,
                                    137, 147, 148, 152, 153,
                                    154, 155, 156, 158, 162,
                                    163, 165])

    sounds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    normalize = False
    time_window = 0.1  # time window in sec

    # filtering parameters
    normalization_type = 'sum_1'
    sigma_smoothing = 0
    sigma_gauss_norm = 1

    # use the mean subtracted map as the learned map
    mean_subtracted_map = True

    ear = 'ipsi'
    elevations = np.arange(0, 50, 1)
    ########################################################################
    ########################################################################

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
            [x_mono, y_mono, x_mono_mean, y_mono_mean, x_bin,
                y_bin, x_bin_mean, y_bin_mean] = pickle.load(f)

        # define which elevations should be used
        x_mono = x_mono[:, :, elevations, :]
        y_mono = y_mono[:, :, elevations]
        x_mono_mean = x_mono_mean[:, :, elevations, :]
        y_mono_mean = y_mono_mean[:, :, elevations]
        x_bin = x_bin[:, :, elevations, :]
        y_bin = y_bin[:, :, elevations]
        x_bin_mean = x_bin_mean[:, :, elevations, :]
        y_bin_mean = y_bin_mean[:, :, elevations]

        fig = plt.figure(figsize=(20, 5))
        # plt.suptitle('Single Participant')
        # Monoaural Data (Ipsilateral), No Mean Subtracted
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        # plot regression line for each sound
        for i_sound, sound in enumerate(sounds):
            hp.plot_localization_result(x_mono[:, i_sound], y_mono[:, i_sound], ax1, SOUND_FILES,
                                        scale_values=True, linear_reg=True, scatter_data=False)
            ax1.set_title('Monoaural')
            hp.set_axis(ax1)
            ax1.set_ylabel('Estimated Elevation [deg]')
            ax1.set_xlabel('True Elevation [deg]')

            # Monoaural Data (Ipsilateral), Mean Subtracted
            hp.plot_localization_result(x_mono_mean[:, i_sound], y_mono_mean[:, i_sound], ax2, SOUND_FILES,
                                        scale_values=True, linear_reg=True, scatter_data=False)
            ax2.set_title('Mono - Mean')
            hp.set_axis(ax2)
            ax2.set_xlabel('True Elevation [deg]')

            # Binaural Data (Ipsilateral), No Mean Subtracted

            hp.plot_localization_result(x_bin[:, i_sound], y_bin[:, i_sound], ax3, SOUND_FILES,
                                        scale_values=True, linear_reg=True, scatter_data=False)
            ax3.set_title('Binaural')
            hp.set_axis(ax3)
            ax3.set_xlabel('True Elevation [deg]')

            # Binaural Data (Ipsilateral), Mean Subtracted

            hp.plot_localization_result(x_bin_mean[:, i_sound], y_bin_mean[:, i_sound], ax4, SOUND_FILES,
                                        scale_values=True, linear_reg=True, scatter_data=False)
            ax4.set_title('Bin - Mean')
            hp.set_axis(ax4)
            ax4.set_xlabel('True Elevation [deg]')

        # plot a common regression line
        x_mono_ = np.reshape(x_mono, (x_mono.shape[0] * x_mono.shape[1], x_mono.shape[2], x_mono.shape[3]))
        y_mono_ = np.reshape(y_mono, (y_mono.shape[0] * y_mono.shape[1], y_mono.shape[2]))

        x_mono_mean_ = np.reshape(x_mono_mean, (x_mono_mean.shape[0] * x_mono_mean.shape[1], x_mono_mean.shape[2], x_mono_mean.shape[3]))
        y_mono_mean_ = np.reshape(y_mono_mean, (y_mono_mean.shape[0] * y_mono_mean.shape[1], y_mono_mean.shape[2]))

        x_bin_ = np.reshape(x_bin, (x_bin.shape[0] * x_bin.shape[1], x_bin.shape[2], x_bin.shape[3]))
        y_bin_ = np.reshape(y_bin, (y_bin.shape[0] * y_bin.shape[1], y_bin.shape[2]))

        x_bin_mean_ = np.reshape(x_bin_mean, (x_bin_mean.shape[0] * x_bin_mean.shape[1], x_bin_mean.shape[2], x_bin_mean.shape[3]))
        y_bin_mean_ = np.reshape(y_bin_mean, (y_bin_mean.shape[0] * y_bin_mean.shape[1], y_bin_mean.shape[2]))

        hp.plot_localization_result(x_mono_, y_mono_, ax1, SOUND_FILES, scale_values=False, linear_reg=True,
                                    disp_values=True, scatter_data=False, reg_color="black")
        hp.plot_localization_result(x_mono_mean_, y_mono_mean_, ax2, SOUND_FILES, scale_values=False,
                                    linear_reg=True, disp_values=True, scatter_data=False, reg_color="black")
        hp.plot_localization_result(x_bin_, y_bin_, ax3, SOUND_FILES, scale_values=False, linear_reg=True,
                                    disp_values=True, scatter_data=False, reg_color="black")
        hp.plot_localization_result(x_bin_mean_, y_bin_mean_, ax4, SOUND_FILES, scale_values=False,
                                    linear_reg=True, disp_values=True, scatter_data=False, reg_color="black")

        # get the name of the sounds
        sound_names = np.array([i.name.split('.')[0] for i in SOUND_FILES])

        lgd = ax4.legend(sound_names, loc=(1.04, 0))

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name / exp_name_str
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            logger.info('Saving figures to ' + fig_save_path.as_posix())
            plt.savefig((fig_save_path / (exp_name + '_localization_sounds.' + save_type)).as_posix(),
                        dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.show()
    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
