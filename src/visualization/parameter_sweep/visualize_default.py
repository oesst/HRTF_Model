import matplotlib.pyplot as plt
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

# Parameters
@click.command()
@click.option('--save_figs', type=click.BOOL, default=False, help='Save figures')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
@click.option('--model_name', default='parameter_sweep', help='Defines the model name.')
@click.option('--exp_name', default='default', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--snr', default=0.2, help='Signal to noise ration to use. Default is 0.2')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--clean', is_flag=True)
def main(save_figs=False, save_type='svg', model_name='parameter_sweep', exp_name='default', azimuth=12, snr=0.2, freq_bands=128, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', clean=False):

    logger = logging.getLogger(__name__)
    logger.info('Showing parameter sweep results')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################

    # make sure save type is given
    if not save_type or len(save_type) == 0:
        save_type = 'svg'

    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, elevations, 1)

    sigma_smoothing_vals = np.arange(0.1, 3.0, 0.05)
    sigma_gauss_norm_vals = np.arange(0.1, 3.0, 0.05)
    ########################################################################
    ########################################################################

    # create unique experiment name
    exp_name_str = hp.create_exp_name([exp_name, normalization_type, mean_subtracted_map, time_window, int(
        snr * 100), freq_bands, max_freq, (azimuth - 12) * 10, normalize, len(elevations), ear])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [scores] = pickle.load(f)

        fig = plt.figure(figsize=(20, 5))
        axes = fig.subplots(1,3,sharex=True,sharey=True)
        ax = axes[0]
        p = ax.pcolormesh(sigma_smoothing_vals, sigma_gauss_norm_vals, np.squeeze(scores[:, :, 0]), vmin=-0.1, vmax=1.0)
        ax.set_xlabel(r'Gauss Normalization $\sigma$')
        ax.set_ylabel(r'Smoothing $\sigma$')
        # ax.set_title('Gain')
        cbar = plt.colorbar(p,ax=ax)
        cbar.set_label('Gain', rotation=270)
        cbar.ax.get_yaxis().labelpad = 15

        ax = axes[1]
        p = ax.pcolormesh(sigma_smoothing_vals, sigma_gauss_norm_vals, np.squeeze(scores[:, :, 1]))
        ax.set_xlabel(r'Gauss Normalization $\sigma$')
        # ax.set_ylabel('Smoothing Sigma')
        # ax.set_title('Bias')
        cbar = plt.colorbar(p,ax=ax)
        cbar.set_label('Bias', rotation=270)
        cbar.ax.get_yaxis().labelpad = 15

        ax = axes[2]
        p = ax.pcolormesh(sigma_smoothing_vals, sigma_gauss_norm_vals, np.squeeze(scores[1:, 1:, 2]), vmin=0.0)
        ax.set_xlabel(r'Gauss Normalization $\sigma$')
        # ax.set_ylabel('Smoothing Sigma')
        # ax.set_title('Score')
        cbar = plt.colorbar(p,ax=ax)
        cbar.set_label('Score', rotation=270)
        cbar.ax.get_yaxis().labelpad = 15

        plt.tight_layout()

        if save_figs:
            fig_save_path = ROOT / 'reports' / 'figures' / model_name / exp_name_str
            if not fig_save_path.exists():
                fig_save_path.mkdir(parents=True, exist_ok=True)
            logger.info('Saving figures to ' + fig_save_path.as_posix())
            plt.savefig((fig_save_path / (exp_name + '_sweep.' + save_type)).as_posix(), dpi=300)
        else:
            plt.show()
    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
