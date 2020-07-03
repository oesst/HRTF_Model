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
    x, y = hp_vis.scale_v(x, y,x.shape[1])
    x = np.reshape(x, (x.shape[0] * x.shape[1], 2))
    y = np.reshape(y, (y.shape[0] * y.shape[1]))

    lr_model = hp_vis.LinearReg(x, y)
    c_m, c_b = lr_model.get_coefficients()
    score = lr_model.get_score()

    return c_m, c_b, score


# Parameters
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
def main(save_figs=False, save_type='svg', model_name='all_participants', exp_name='localization_default', azimuth=12, snr=0.2, freq_bands=24, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1, clean=False):

    logger = logging.getLogger(__name__)
    logger.info('Showing localization results for all participants')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################

    # make sure save type is given
    if not save_type or len(save_type) == 0:
        save_type = 'svg'

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

    exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map, time_window, int(
        snr * 100), freq_bands, max_freq, (azimuth - 12) * 10, normalize, len(elevations), ear])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if not clean and exp_path.exists() and exp_file.is_file():
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
        fig = plt.figure(figsize=(20, 5))
        my_pal = [(	31/255, 119/255, 180/255), (0.3333333333333333, 0.6588235294117647, 0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
                  (0.5058823529411764, 0.4470588235294118, 0.6980392156862745)]

        ax = fig.add_subplot(1, 3, 1)
        ax.set_ylabel('Gain')
        sns.boxplot(data=coeff_ms.T, showfliers=True, palette=hp_vis.MY_COLORS, ax=ax, linewidth=3)
        ax.set_xticklabels(['Mono', 'Mono\n-Prior', 'Bin', 'Bin\n-Prior'])

        ax = fig.add_subplot(1, 3, 2)
        ax.set_ylabel('Bias')
        sns.boxplot(data=coeff_bs.T, showfliers=True, palette=hp_vis.MY_COLORS, ax=ax, linewidth=3)
        ax.set_xticklabels(['Mono', 'Mono\n-Prior', 'Bin', 'Bin\n-Prior'])

        ax = fig.add_subplot(1, 3, 3)
        ax.set_ylabel('Score')
        sns.boxplot(data=scores.T, showfliers=True, palette=hp_vis.MY_COLORS, ax=ax, linewidth=3)
        ax.set_xticklabels(['Mono', 'Mono\n-Prior', 'Bin', 'Bin\n-Prior'])

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name / exp_name_str
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
