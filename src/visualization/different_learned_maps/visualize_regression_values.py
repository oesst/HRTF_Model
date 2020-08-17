import matplotlib.pyplot as plt
import src.features.helpers_vis as hp_vis
import src.features.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click
import seaborn as sns

hp_vis.set_layout(15)

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))


def get_regression_values(x, y):
    x, y = hp_vis.scale_v(x, y, x.shape[1])
    x = np.reshape(x, (x.shape[0] * x.shape[1], 2))
    y = np.reshape(y, (y.shape[0] * y.shape[1]))

    lr_model = hp_vis.LinearReg(x, y)
    c_m, c_b = lr_model.get_coefficients()
    score = lr_model.get_score()

    return c_m, c_b, score


# Define whether figures should be saved
@click.command()
@click.option('--save_figs', type=click.BOOL, default=False, help='Save figures')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
@click.option('--model_name', default='single_participant', help='Defines the model name.')
@click.option('--exp_name', default='single_participant_default', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--snr', default=0.2, help='Signal to noise ration to use. Default is 0.2')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--sigma_smoothing', default=0, help='Sigma for smoothing kernel. 0 is off. Default is 0.')
@click.option('--sigma_gauss_norm', default=1, help='Sigma for gauss normalization. 0 is off. Default is 1.')
@click.option('--clean', is_flag=True)
def main(save_figs=False, save_type='svg', model_name='different_learned_maps', exp_name='localization_default', azimuth=12, snr=0.2, freq_bands=24, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1, clean=False):

    logger = logging.getLogger(__name__)
    logger.info('Showing localization results for all participants')

    # make sure save type is given
    if not save_type or len(save_type) == 0:
        save_type = 'svg'

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################

    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
                                    12, 15, 17, 18, 19, 20,
                                    21, 27, 28, 33, 40, 44,
                                    48, 50, 51, 58, 59, 60,
                                    61, 65, 119, 124, 126,
                                    127, 131, 133, 134, 135,
                                    137, 147, 148, 152, 153,
                                    154, 155, 156, 158, 162,
                                    163, 165])

    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, elevations, 1)

    ########################################################################
    ########################################################################

    # create unique experiment name
    exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map,
                                       time_window, int(snr * 100), freq_bands, max_freq, (azimuth - 12) * 10, normalize, len(elevations), ear])

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
        x_mono_all = x_mono[:, :, :, elevations, :]
        y_mono_all = y_mono[:, :, :, elevations]
        x_mono_mean_all = x_mono_mean[:, :, :, elevations, :]
        y_mono_mean_all = y_mono_mean[:, :, :, elevations]
        x_bin_all = x_bin[:, :, :, elevations, :]
        y_bin_all = y_bin[:, :, :, elevations]
        x_bin_mean_all = x_bin_mean[:, :, :, elevations, :]
        y_bin_mean_all = y_bin_mean[:, :, :, elevations]

        fig = plt.figure(figsize=(20, 20))
        axes = fig.subplots(4, 3, sharex=True, sharey=False)

        # plot regression line for each participant
        for i_map in range(4):

            # get right axis
            ax1 = axes[i_map, 0]
            ax2 = axes[i_map, 1]
            ax3 = axes[i_map, 2]

            # get the data for each map
            x_mono = x_mono_all[i_map]
            y_mono = y_mono_all[i_map]
            x_mono_mean = x_mono_mean_all[i_map]
            y_mono_mean = y_mono_mean_all[i_map]
            x_bin = x_bin_all[i_map]
            y_bin = y_bin_all[i_map]
            x_bin_mean = x_bin_mean_all[i_map]
            y_bin_mean = y_bin_mean_all[i_map]

            # save regression values for later usage
            coeff_ms = np.zeros((4, participant_numbers.shape[0]))
            coeff_bs = np.zeros((4, participant_numbers.shape[0]))
            scores = np.zeros((4, participant_numbers.shape[0]))

            # plot regression line for each participant
            for i_par, par in enumerate(participant_numbers):
                coeff_ms[0, i_par], coeff_bs[0, i_par], scores[0, i_par] = get_regression_values(x_mono[i_par], y_mono[i_par])
                coeff_ms[1, i_par], coeff_bs[1, i_par], scores[1, i_par] = get_regression_values(x_mono_mean[i_par], y_mono_mean[i_par])
                coeff_ms[2, i_par], coeff_bs[2, i_par], scores[2, i_par] = get_regression_values(x_bin[i_par], y_bin[i_par])
                coeff_ms[3, i_par], coeff_bs[3, i_par], scores[3, i_par] = get_regression_values(x_bin_mean[i_par], y_bin_mean[i_par])

                # sns.set_palette('muted')
                # my_pal = sns.color_palette("hls", 8)

            if i_map == 0:
                ax1.set_title('Monoaural', rotation='vertical', x=-0.2, y=0.7)
            elif i_map == 1:
                ax1.set_title('Mono - Prior', rotation='vertical', x=-0.2, y=0.8)
            elif i_map == 2:
                ax1.set_title('Binaural', rotation='vertical', x=-0.2, y=0.6)
            else:
                ax1.set_title('Bin - Prior', rotation='vertical', x=-0.2, y=0.7)

            ax1.set_ylabel('Gain')
            sns.boxplot(data=coeff_ms.T, showfliers=True, palette=hp_vis.MY_COLORS, ax=ax1, linewidth=3)
            ax1.set_ylim([0, 1])

            ax2.set_ylabel('Bias')
            sns.boxplot(data=coeff_bs.T, showfliers=True, palette=hp_vis.MY_COLORS, ax=ax2, linewidth=3)
            # ax2.set_ylim([0,20])


            ax3.set_ylabel('Score')
            sns.boxplot(data=scores.T, showfliers=True, palette=hp_vis.MY_COLORS, ax=ax3, linewidth=3)
            ax3.set_ylim([0, 1])
            if i_map == 3:
                ax3.set_xticklabels(['Mono', 'Mono\n-Mean', 'Bin', 'Bin\n-Mean'])

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / exp_name_str / model_name
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            logger.info('Saving figures to ' + fig_save_path.as_posix())
            plt.savefig((fig_save_path / (model_name + '_' + exp_name + '_regression_values.' + save_type)).as_posix(), dpi=300)

        else:
            plt.show()

    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
