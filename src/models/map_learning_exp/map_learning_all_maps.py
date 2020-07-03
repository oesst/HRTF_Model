# -*- coding: utf-8 -*-
import logging
import click
from pathlib import Path
from src.data import generateData
from src.features import helpers as hp
# from src.visualization import helpers as hpVis
import numpy as np
import pickle
import random

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))

# Define whether figures should be saved
@click.command()
@click.option('--model_name', default='map_learning', help='Defines the model name.')
@click.option('--exp_name', default='localization_all_maps', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--snr', default=0.2, help='Signal to noise ration to use. Default is 0.2')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--n_trials', default=500, help='Number of learning n_trials. Default 100')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--sigma_smoothing', default=0, help='Sigma for smoothing kernel. 0 is off. Default is 0.')
@click.option('--sigma_gauss_norm', default=1, help='Sigma for gauss normalization. 0 is off. Default is 1.')
@click.option('--clean', is_flag=True)
def main(model_name = 'map_learning', exp_name = 'localization_all_maps', azimuth=12, snr=0.2, freq_bands=24, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', n_trials=100, normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1, clean=False):
    """ Learns the elevation spectra map gradually over presented sounds and saves the localization quality for each trial
    """
    logger = logging.getLogger(__name__)
    logger.info('Learning different maps for all participants')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    # participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
    #                                 12, 15, 17, 18, 19, 20, 21, 27, 28, 33, 40])

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
    exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map, time_window, int(
        snr * 100), freq_bands, max_freq, (azimuth - 12) * 10, normalize, len(elevations), ear, n_trials])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if not clean and exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with exp_file.open('rb') as f:
            logger.info('Reading model data from file')
            [mono_res, mono_mean_res, bin_res, bin_mean_res, trial_used_ss] = pickle.load(f)
    else:

        # store only the localization coefficeints (gain,bias,score)
        mono_res = np.zeros((4,len(participant_numbers), n_trials, 3))
        mono_mean_res = np.zeros((4,len(participant_numbers), n_trials, 3))
        bin_res = np.zeros((4,len(participant_numbers), n_trials, 3))
        bin_mean_res = np.zeros((4,len(participant_numbers), n_trials, 3))
        trial_used_ss = np.zeros((4,len(participant_numbers), n_trials))

        # learned_maps_participants = np.zeros((len(participant_numbers), len(elevations), freq_bands))

        for i_par, par in enumerate(participant_numbers):
            logger.info('Localizing {0:d} trials for participant {1:d}. \n'.format(n_trials,par))

            # create or read the data. psd_all_c = (sounds,elevations,frequency bands)
            psd_all_c, psd_all_i = generateData.create_data(
                freq_bands, par, snr, normalize, azimuth, time_window, max_freq=max_freq)

            # Take only given elevations
            psd_all_c = psd_all_c[:, elevations, :]
            psd_all_i = psd_all_i[:, elevations, :]

            # filter data and integrate it
            psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = hp.process_inputs(
                psd_all_i, psd_all_c, ear, normalization_type, sigma_smoothing, sigma_gauss_norm)

            # walk over test n_trials, in this case the number of sound samples

            for i_trials in range(n_trials):

                # decide how many sound samples should be used for the map. this is between 1 and number_of_sounds * number_of_elevations
                number_of_ss = np.random.randint(1, psd_all_c.shape[0] * psd_all_c.shape[1])
                # choose the sound samples to learn the map
                ind = np.random.randint(0, high=(psd_all_c.shape[0] * psd_all_c.shape[1]), size=number_of_ss)
                # get the indices for the sound_inds
                sounds_ind = np.unravel_index(ind, (psd_all_c.shape[0], psd_all_c.shape[1]))

                # decide which type of map is learned_map
                for i_maps in range(4):

                    # monaural condition
                    if i_maps == 0:
                        # get only the defined sounds and elevations
                        tmp_data = np.zeros(psd_binaural.shape)
                        tmp_data[sounds_ind[0], sounds_ind[1], :] = psd_mono[sounds_ind[0], sounds_ind[1], :]
                        # create learned_map
                        learned_map = hp.create_map(tmp_data, False)
                    elif i_maps == 1:
                        # get only the defined sounds and elevations
                        tmp_data = np.zeros(psd_binaural.shape)
                        tmp_data[sounds_ind[0], sounds_ind[1], :] = psd_mono[sounds_ind[0], sounds_ind[1], :]
                        # create learned_map
                        learned_map = hp.create_map(tmp_data, True)
                    elif i_maps == 2:
                        # get only the defined sounds and elevations
                        tmp_data = np.zeros(psd_binaural.shape)
                        tmp_data[sounds_ind[0], sounds_ind[1], :] = psd_binaural[sounds_ind[0], sounds_ind[1], :]
                        # create learned_map
                        learned_map = hp.create_map(tmp_data, False)
                    elif i_maps == 3:
                        # get only the defined sounds and elevations
                        tmp_data = np.zeros(psd_binaural.shape)
                        tmp_data[sounds_ind[0], sounds_ind[1], :] = psd_binaural[sounds_ind[0], sounds_ind[1], :]
                        # create learned_map
                        learned_map = hp.create_map(tmp_data, True)

                    # store the map
                    # learned_maps_participants[i_par, :, :] = learned_map
                    # store the number of sounds used
                    trial_used_ss[i_maps, i_par, i_trials] = number_of_ss

                    # localize the sounds and save the results
                    x, y = hp.localize_sound(psd_mono, learned_map)
                    mono_res[i_maps, i_par, i_trials, :] = hp.get_localization_coefficients_score(x, y)
                    # localize the sounds and save the results
                    x, y = hp.localize_sound(psd_mono_mean, learned_map)
                    mono_mean_res[i_maps, i_par, i_trials, :] = hp.get_localization_coefficients_score(x, y)

                    # localize the sounds and save the results
                    x, y = hp.localize_sound(psd_binaural, learned_map)
                    bin_res[i_maps, i_par, i_trials, :] = hp.get_localization_coefficients_score(x, y)

                    # localize the sounds and save the results
                    x, y = hp.localize_sound(psd_binaural_mean, learned_map)
                    bin_mean_res[i_maps, i_par, i_trials, :] = hp.get_localization_coefficients_score(x, y)

        # create Path
        exp_path.mkdir(parents=True, exist_ok=True)
        with exp_file.open('wb') as f:
            logger.info('Creating model file')
            pickle.dump([mono_res, mono_mean_res, bin_res, bin_mean_res, trial_used_ss], f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
