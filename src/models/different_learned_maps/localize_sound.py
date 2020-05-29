# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from src.data import generateData
from src.features import helpers as hp
# from src.visualization import helpers as hpVis
import numpy as np
import pickle

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / 'data/raw/sound_samples/'
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob('**/*.wav'))


def main(model_name='different_learned_maps', exp_name='localization_default'):
    """ This script takes the filtered data and tries to localize sounds with different, learned map
        for all participants.
    """
    logger = logging.getLogger(__name__)
    logger.info('Localizing sounds for all participants, different maps')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    azimuth = 12
    snr = 0.2
    freq_bands = 128
    max_freq = 22000
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

    # filtering parameters
    normalization_type = 'sum_1'
    sigma_smoothing = 0
    sigma_gauss_norm = 1

    # use the mean subtracted map as the learned map
    mean_subtracted_map = True

    ear = 'contra'

    elevations = np.arange(0, 25, 1)
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
        with exp_file.open('rb') as f:
            logger.info('Reading model data from file')
            [x_mono, y_mono, x_mono_mean, y_mono_mean, x_bin,
                y_bin, x_bin_mean, y_bin_mean] = pickle.load(f)
    else:

        x_mono = np.zeros((4, len(participant_numbers), len(SOUND_FILES), len(elevations), 2))
        y_mono = np.zeros((4, len(participant_numbers), len(SOUND_FILES), len(elevations)))
        x_mono_mean = np.zeros((4, len(participant_numbers), len(SOUND_FILES), len(elevations), 2))
        y_mono_mean = np.zeros((4, len(participant_numbers), len(SOUND_FILES), len(elevations)))
        x_bin = np.zeros((4, len(participant_numbers), len(SOUND_FILES), len(elevations), 2))
        y_bin = np.zeros((4, len(participant_numbers), len(SOUND_FILES), len(elevations)))
        x_bin_mean = np.zeros((4, len(participant_numbers), len(SOUND_FILES), len(elevations), 2))
        y_bin_mean = np.zeros((4, len(participant_numbers), len(SOUND_FILES), len(elevations)))
        for i_par, par in enumerate(participant_numbers):

            # create or read the data
            psd_all_c, psd_all_i = generateData.create_data(
                freq_bands, par, snr, normalize, azimuth, time_window,max_freq=max_freq)

            # Take only given elevations
            psd_all_c = psd_all_c[:, elevations, :]
            psd_all_i = psd_all_i[:, elevations, :]

            # filter data and integrate it
            psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = hp.process_inputs(
                psd_all_i, psd_all_c, ear, normalization_type, sigma_smoothing, sigma_gauss_norm)

            # walk over the 4 different maps: mono, mono_mean, bina, bina_mean
            for i_map in range(4):
                # create map from defined processed data

                if i_map == 0:
                    learned_map = hp.create_map(psd_mono, False)
                elif i_map == 1:
                    learned_map = hp.create_map(psd_mono, True)
                elif i_map == 2:
                    learned_map = hp.create_map(psd_binaural, False)
                elif i_map == 3:
                    # bina_mean
                    learned_map = hp.create_map(psd_binaural, True)
                else:
                    logger.error('Something went wrong in if i_map statement')

                # localize the sounds and save the results
                x_mono[i_map, i_par, :, :, :], y_mono[i_map, i_par, :] = hp.localize_sound(psd_mono, learned_map)

                # localize the sounds and save the results
                x_mono_mean[i_map, i_par, :, :, :], y_mono_mean[i_map, i_par, :, :] = hp.localize_sound(psd_mono_mean, learned_map)

                # localize the sounds and save the results
                x_bin[i_map, i_par, :, :, :], y_bin[i_map, i_par, :, :] = hp.localize_sound(psd_binaural, learned_map)

                # localize the sounds and save the results
                x_bin_mean[i_map, i_par, :, :, :], y_bin_mean[i_map, i_par, :, :] = hp.localize_sound(psd_binaural_mean, learned_map)

        # create Path
        exp_path.mkdir(parents=True, exist_ok=True)
        with exp_file.open('wb') as f:
            logger.info('Creating model file')
            pickle.dump([x_mono, y_mono, x_mono_mean, y_mono_mean,
                         x_bin, y_bin, x_bin_mean, y_bin_mean], f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
