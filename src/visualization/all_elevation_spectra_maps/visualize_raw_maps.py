import matplotlib.pyplot as plt
import matplotlib as mpl
import src.features.filters as filters
import src.visualization.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click





hp.set_layout(15)


ROOT = Path(__file__).resolve().parents[3]
# ROOT = Path('.').resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))

# Define whether figures should be saved
@click.command()
@click.option('--save_figs', default=False, help='Save the figures.')
@click.option('--save_type', default='svg', help='Define the format figures are saved.')
def main(save_figs=False, save_type='svg', model_name='elevation_spectra_maps', exp_name='unfiltered'):

    logger = logging.getLogger(__name__)
    logger.info('Showing localization results for all participants')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    azimuth = 12
    snr = 0.2
    freq_bands = 128
    max_freq = 20000
    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
                                    12, 15, 17, 18, 19, 20, 21, 27, 28, 33, 40])

    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, 25, 1)
    ########################################################################
    ########################################################################

    # create unique experiment name
    exp_name_str = exp_name + '_' + str(time_window) + '_window_' + str(
        int(snr * 100)) + '_srn_' + str(freq_bands) + '_channels_'+str(max_freq)+'_max_freq_' + str((azimuth - 12) * 10) + '_azi_' + str(normalize) + '_norm' + str(len(elevations)) + '_elevs.npy'
    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str

    # set tick labels

    # if len(elevations) == 50:
    #
    #     t = np.zeros(8)
    #     t[0] = -55
    #     t[1] = -45
    #     t[2] = -45 + 1 * 68
    #     t[3] = -45 + 2 * 68
    #     t[4] = -45 + 3 * 68
    #     t[5] = -45 + 4 * 68
    #     t[6] = 230
    #     t[7] = 230
    # else:
    #     t = np.zeros(8)
    #     t[0] = -55
    #     t[1] = -45
    #     t[2] = -45 + 1 * 27
    #     t[3] = -45 + 2 * 27
    #     t[4] = -45 + 3 * 27
    #     t[5] = -45 + 4 * 27
    #     t[6] = -45 + 5 * 27
    #     t[7] = 100

    fig_size = (7, 5)
    # fig_size = (20, 14)

    formatter = hp.ERBFormatter(100, 18000, unit='', places=0)


    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), 'rb') as f:
            logger.info('Reading model data from file')
            [ipsi_maps, contra_maps] = pickle.load(f)

        ipsi_maps = ipsi_maps[:, :, elevations, :]
        contra_maps = contra_maps[:, :, elevations, :]

        # plot regression line for each participant
        for i_par, par in enumerate(participant_numbers):

            for i_sound, sound in enumerate(SOUND_FILES):
                sound = sound.name.split('.')[0]
                # CONTRA
                fig = plt.figure(figsize=fig_size)
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(sound)
                # ax.imshow(np.squeeze(ipsi_maps[i_par, i_sound]),interpolation = 'bilinear')
                data = np.squeeze(ipsi_maps[i_par, i_sound])
                # ax.pcolormesh(np.squeeze(ipsi_maps[i_par, i_sound]),shading='gouraud',linewidth=0,rasterized=True)
                c = ax.pcolormesh(np.linspace(0, 1, data.shape[1]), np.linspace(-45, 90, data.shape[0]),
                              data, shading='gouraud', linewidth=0, rasterized=True)
                plt.colorbar(c)
                ax.xaxis.set_major_formatter(formatter)
                ax.set_xlabel('Frequency [Hz]')
                ax.set_ylabel('Elevations [deg]')
                # ax.set_yticklabels(t[1:-1])

                if save_figs:
                    fig_save_path = ROOT / 'reports' / 'figures' / model_name / ('participant_' + str(par))
                    if not fig_save_path.exists():
                        fig_save_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig((fig_save_path / (exp_name + '_raw_maps_ipsi_' + str(sound) + '.' + save_type)).as_posix(), dpi=300, transparent=True)
                    plt.close()

                else:
                    plt.show()

                # CONTRA
                fig = plt.figure(figsize=fig_size)
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(sound)
                # ax.pcolormesh(np.squeeze(contra_maps[i_par, i_sound]), shading='gouraud', linewidth=0, rasterized=True)
                data = np.squeeze(contra_maps[i_par, i_sound])
                # ax.pcolormesh(np.squeeze(ipsi_maps[i_par, i_sound]),shading='gouraud',linewidth=0,rasterized=True)
                c = ax.pcolormesh(np.linspace(0, 1, data.shape[1]), np.linspace(-45, 90, data.shape[0]),
                              data, shading='gouraud', linewidth=0, rasterized=True)
                plt.colorbar(c)
                ax.xaxis.set_major_formatter(formatter)
                ax.set_xlabel('Frequency [Hz]')
                ax.set_ylabel('Elevations [deg]')
                # ax.set_yticklabels(t[1:-1])

                if save_figs:
                    fig_save_path = ROOT / 'reports' / 'figures' / model_name / exp_name_str / ('participant_' + str(par))
                    if not fig_save_path.exists():
                        fig_save_path.mkdir(parents=True, exist_ok=True)
                    plt.savefig((fig_save_path / (exp_name + '_raw_maps_contra_' + str(sound) +
                                                  '.' + save_type)).as_posix(), dpi=300, transparent=True)
                    plt.close()
                else:
                    plt.show()

    else:
        logger.error('No data set found. Run model first!')
        logger.error(exp_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
