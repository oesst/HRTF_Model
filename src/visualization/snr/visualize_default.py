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

# Parameters
@click.command()
@click.option('--save_figs', type=click.BOOL, default=False, help='Save figures')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
@click.option('--model_name', default='snr_experiment', help='Defines the model name.')
@click.option('--exp_name', default='default', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--snr', default=0.2, help='Signal to noise ration to use. Default is 0.2')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--sigma_smoothing', default=0, help='Sigma for smoothing kernel. 0 is off. Default is 0.')
@click.option('--sigma_gauss_norm', default=1, help='Sigma for gauss normalization. 0 is off. Default is 1.')
@click.option('--clean', is_flag=True)
def main(save_figs=False, save_type='svg', model_name='all_participants', exp_name='localization_default', azimuth=12, freq_bands=24, max_freq=20000, elevations=25, snr=0.2, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1, clean=False):

    logger = logging.getLogger(__name__)
    logger.info('Showing localization results for all participants')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################

    # make sure save type is given
    if not save_type or len(save_type) == 0:
        save_type = 'svg'

    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, elevations, 1)

    snrs = np.arange(0.0, 1.1, 0.1)
    ########################################################################
    ########################################################################

    # create unique experiment name
    exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map,
                                       time_window, freq_bands, max_freq, (azimuth - 12) * 10, normalize, len(elevations), ear])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [scores] = pickle.load(f)

        snrs_all = np.tile(snrs, (scores.shape[0], 1))
        snrs_all = np.reshape(snrs_all, (snrs_all.shape[0] * snrs_all.shape[1]))
        scores_tmp = np.reshape(scores, (scores.shape[0] * scores.shape[1], scores.shape[2], scores.shape[3]))
        fig = plt.figure(figsize=(20, 7))

        ax = fig.add_subplot(1, 3, 1)
        y = scores_tmp[:, 0, 0]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C0, label='Mono')

        y = scores_tmp[:, 1, 0]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C1, label='Mono-Mean')

        y = scores_tmp[:, 2, 0]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C2, label='Bin')

        y = scores_tmp[:, 3, 0]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C3, label='Bin-Mean')
        ax.set_ylabel('Gain')
        ax.set_xlabel('SNR')
        ax.set_ylim([0.0, 1.1])

        ax = fig.add_subplot(1, 3, 2)
        y = scores_tmp[:, 0, 1]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C0, label='Mono')

        y = scores_tmp[:, 1, 1]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C1, label='Mono-Mean')

        y = scores_tmp[:, 2, 1]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C2, label='Bin')

        y = scores_tmp[:, 3, 1]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C3, label='Bin-Mean')
        ax.set_ylabel('Bias')
        ax.set_xlabel('SNR')
        ax.set_ylim([0.0, 30])

        ax = fig.add_subplot(1, 3, 3)
        y = scores_tmp[:, 0, 2]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C0, label='Monaural')

        y = scores_tmp[:, 1, 2]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C1, label='Mono-Mean')

        y = scores_tmp[:, 2, 2]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C2, label='Binaural')

        y = scores_tmp[:, 3, 2]
        sns.regplot(x=snrs_all, y=y, x_estimator=np.mean, order=2, ax=ax, color=hp_vis.C3, label='Bin-Mean')
        ax.set_ylabel('Score')
        ax.set_xlabel('SNR')
        ax.set_ylim([0.0, 1.4])

        lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.35, 0.8))

        plt.tight_layout()

        if save_figs:
            exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map, time_window, int(
            snr * 100), freq_bands, max_freq, (azimuth - 12) * 10, normalize, len(elevations), ear])
            fig_save_path = ROOT / 'reports' / 'figures' / exp_name_str / model_name
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            logger.info('Saving figures to ' + fig_save_path.as_posix())
            plt.savefig((fig_save_path / (model_name + '_' + exp_name + '_localization.' + save_type)).as_posix(), dpi=300)
        else:
            plt.show()
    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
