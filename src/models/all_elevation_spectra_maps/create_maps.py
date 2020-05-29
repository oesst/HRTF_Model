# -*- coding: utf-8 -*-
import logging
import click
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


def process_inputs(psd_all_i, psd_all_c, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1):
    # filter the data
    psd_mono_c = hp.filter_dataset(psd_all_c, normalization_type=normalization_type,
                                   sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)
    psd_mono_i = hp.filter_dataset(psd_all_i, normalization_type=normalization_type,
                                   sigma_smoothing=sigma_smoothing, sigma_gauss_norm=sigma_gauss_norm)

    # integrate the signals and filter
    if ear.find('contra') >= 0:
        psd_binaural = hp.filter_dataset(
            psd_mono_c / psd_mono_i, normalization_type=normalization_type, sigma_smoothing=0, sigma_gauss_norm=0)
    else:
        psd_binaural = hp.filter_dataset(
            psd_mono_i / psd_mono_c, normalization_type=normalization_type, sigma_smoothing=0, sigma_gauss_norm=0)

    # calculate different input sounds. should be 4 of them (mono,mono-mean,bin, bin-mean)
    if ear.find('contra') >= 0:
        psd_mono = psd_mono_c
    else:
        psd_mono = psd_mono_i

    psd_mono_mean = psd_mono - \
        np.transpose(np.tile(np.mean(psd_mono, axis=1), [
                     psd_mono.shape[1], 1, 1]), [1, 0, 2])
    psd_binaural = psd_binaural
    psd_binaural_mean = psd_binaural - \
        np.transpose(np.tile(np.mean(psd_binaural, axis=1), [
                     psd_binaural.shape[1], 1, 1]), [1, 0, 2])

    return psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean



@click.command()
@click.option('--model_name', default='single_participant', help='Defines the model name.')
@click.option('--exp_name', default='single_participant_default', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--participant_numbers', help='CIPIC participant number. Default is None')
@click.option('--snr', default=0.2, help='Signal to noise ration to use. Default is 0.2')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--clean', is_flag=True)
def main(model_name='elevation_spectra_maps', exp_name='unfiltered', azimuth=12, participant_numbers=None, snr=0.2, freq_bands=24, max_freq=20000, elevations=25, clean=False):
    """ TODO
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating maps for all participants and sounds')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################

    # if participant_numbers is not given we use all of them
    if not participant_numbers:
        participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
                                        12, 15, 17, 18, 19, 20,
                                        21, 27, 28, 33, 40, 44,
                                        48, 50, 51, 58, 59, 60,
                                        61, 65, 119, 124, 126,
                                        127, 131, 133, 134, 135,
                                        137, 147, 148, 152, 153,
                                        154, 155, 156, 158, 162,
                                        163, 165])
    else:
        # participant_numbers are given. need to be cast to int array
        participant_numbers = np.array([int(i) for i in participant_numbers.split(',')])
        print(participant_numbers)

    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, elevations, 1)
    ########################################################################
    ########################################################################

    # create unique experiment name
    exp_name_str = hp.create_exp_name([exp_name, time_window, int(snr * 100), freq_bands, max_freq, participant_numbers, (azimuth - 12) * 10, normalize, len(elevations)])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if not clean and exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with exp_file.open('rb') as f:
            logger.info('Reading model data from file')
            [ipsi_maps,contra_maps] = pickle.load(f)
    else:

        ipsi_maps = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations), freq_bands))
        contra_maps = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations), freq_bands))

        for i_par, par in enumerate(participant_numbers):

            # create or read the data
            psd_all_c, psd_all_i = generateData.create_data(
                freq_bands, par, snr, normalize, azimuth, time_window,max_freq=max_freq)

            # Take only given elevations
            psd_all_c = psd_all_c[:, elevations, :]
            psd_all_i = psd_all_i[:, elevations, :]

            ipsi_maps[i_par, :, :, :] = psd_all_i
            contra_maps[i_par, :, :, :] = psd_all_c

        # create Path
        exp_path.mkdir(parents=True, exist_ok=True)
        with exp_file.open('wb') as f:
            logger.info('Creating model file')
            pickle.dump([ipsi_maps,contra_maps], f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
