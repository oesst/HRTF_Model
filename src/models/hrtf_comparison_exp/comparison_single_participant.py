# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from src.data import generateHRTFs
from src.data import generateData
from src.features import helpers as hp
from src.features import helpers_vis as hpVis
import numpy as np
import pickle
import matplotlib.pyplot as plt

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


def pearson2d(A, B):
    """ Calculate a 2d pearson correlation index """
    A_ = A - A.mean()
    B_ = B - B.mean()
    r = (A_ * B_).sum() / np.sqrt((A_**2).sum() * (B_**2).sum())
    return r


# Define whether figures should be saved
@click.command()
@click.option('--model_name', default='single_participant', help='Defines the model name.')
@click.option('--exp_name', default='single_participant_default', help='Defines the experiment name')
@click.option('--azimuth', default=12, help='Azimuth for which localization is done. Default is 12')
@click.option('--participant_number', default=9, help='CIPIC participant number. Default is 9')
@click.option('--snr', default=0.0, help='Signal to noise ration to use. Default is 0.0')
@click.option('--freq_bands', default=128, help='Amount of frequency bands to use. Default is 128')
@click.option('--max_freq', default=20000, help='Max frequency to use. Default is 20000')
@click.option('--elevations', default=25, help='Number of elevations to use 0-n. Default is 25 which equals 0-90 deg')
@click.option('--mean_subtracted_map', default=True, help='Should the learned map be mean subtracted. Default is True')
@click.option('--ear', default='contra', help='Which ear should be used, contra or ipsi. Default is contra')
@click.option('--normalization_type', default='sum_1', help='Which normalization type should be used sum_1, l1, l2. Default is sum_1')
@click.option('--sigma_smoothing', default=0, help='Sigma for smoothing kernel. 0 is off. Default is 0.')
@click.option('--sigma_gauss_norm', default=1, help='Sigma for gauss normalization. 0 is off. Default is 1.')
@click.option('--clean', is_flag=True)
def main(model_name='hrtf_comparison', exp_name='single_participant', azimuth=12, participant_number=9, snr=0.0, freq_bands=24, max_freq=20000, elevations=25, mean_subtracted_map=True, ear='ipsi', normalization_type='sum_1', sigma_smoothing=0, sigma_gauss_norm=1, clean=False):
    """ This script calculates the correlation coefficient between the ipsi- and contralateral HRTF and the learned maps for a single participant.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        'Comparing learned HRTF maps with the actual HRTF of a participant')

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, elevations, 1)
    ########################################################################
    ########################################################################

    # create unique experiment name
    exp_name_str = hp.create_exp_name([exp_name, normalization_type, sigma_smoothing, sigma_gauss_norm, mean_subtracted_map, time_window, int(
        snr * 100), freq_bands, max_freq, participant_number, (azimuth - 12) * 10, normalize, len(elevations), ear])

    exp_path = ROOT / 'models' / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if not clean and exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with exp_file.open('rb') as f:
            logger.info('Reading model data from file')
            [hrtfs_i, hrtfs_c, learned_map_mono, learned_map_mono_mean, learned_map_bin, learned_map_bin_mean] = pickle.load(f)
    else:

        # create or read the data
        psd_all_c, psd_all_i = generateData.create_data(
            freq_bands, participant_number, snr, normalize, azimuth, time_window,max_freq=max_freq)

        # filter data and integrate it
        psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = process_inputs(
            psd_all_i, psd_all_c, ear, normalization_type, sigma_smoothing, sigma_gauss_norm)

        # create map from defined processed data
        learned_map_mono = hp.create_map(psd_mono, False)
        learned_map_mono_mean = hp.create_map(psd_mono, True)
        learned_map_bin = hp.create_map(psd_binaural, False)
        learned_map_bin_mean = hp.create_map(psd_binaural, True)
        # learned_map = hp.create_map(psd_mono, False)
        # Get the actual HRTF
        hrtfs_c, hrtfs_i = generateHRTFs.create_data(
            freq_bands, participant_number, snr, normalize, azimuth, time_window,max_freq=max_freq)

        # filter data and integrate it
        # hrtfs_c = hp.filter_dataset(hrtfs_c, normalization_type=normalization_type,
        #                                sigma_smoothing=0, sigma_gauss_norm=0)
        hrtfs_i, psd_mono_mean, psd_binaural, psd_binaural_mean = process_inputs(
            hrtfs_i, hrtfs_c, 'ipsi', normalization_type, sigma_smoothing, sigma_gauss_norm)

        hrtfs_c, psd_mono_mean, psd_binaural, psd_binaural_mean = process_inputs(
            hrtfs_i, hrtfs_c, 'contra', normalization_type, sigma_smoothing, sigma_gauss_norm)

        # remove mean for later comparison
        hrtfs_c = np.squeeze(hrtfs_c[0, 0:len(elevations), :])
        hrtfs_c -= hrtfs_c.mean()
        hrtfs_i = np.squeeze(hrtfs_i[0, 0:len(elevations), :])
        hrtfs_i -= hrtfs_i.mean()
        learned_map_mono -= learned_map_mono.mean()
        learned_map_mono_mean -= learned_map_mono_mean.mean()
        learned_map_bin -= learned_map_bin.mean()
        learned_map_bin_mean -= learned_map_bin_mean.mean()

        # ## calculate pearson index
        # correlations[i_par, 0, 0] = pearson2d(learned_map_mono, hrtfs_i)
        # correlations[i_par, 0, 1] = pearson2d(learned_map_mono, hrtfs_c)
        #
        # correlations[i_par, 1, 0] = pearson2d(
        #     learned_map_mono_mean, hrtfs_i)
        # correlations[i_par, 1, 1] = pearson2d(
        #     learned_map_mono_mean, hrtfs_c)
        #
        # correlations[i_par, 2, 0] = pearson2d(learned_map_bin, hrtfs_i)
        # correlations[i_par, 2, 1] = pearson2d(learned_map_bin, hrtfs_c)
        #
        # correlations[i_par, 3, 0] = pearson2d(
        #     learned_map_bin_mean, hrtfs_i)
        # correlations[i_par, 3, 1] = pearson2d(
        #     learned_map_bin_mean, hrtfs_c)

        # create Path
        exp_path.mkdir(parents=True, exist_ok=True)
        with exp_file.open('wb') as f:
            logger.info('Creating model file')
            pickle.dump([hrtfs_i, hrtfs_c, learned_map_mono, learned_map_mono_mean, learned_map_bin, learned_map_bin_mean], f)
    #
    # fig = plt.figure(figsize=(20, 5))
    # # plt.suptitle('Single Participant')
    # # Monoaural Data (Ipsilateral), No Mean Subtracted
    # ax = fig.add_subplot(1, 2, 1)
    # # a = ax.pcolormesh(np.squeeze(hrtfs_i[:,:-5]))
    # data = learned_map_bin[:, :]
    # data = np.squeeze(hrtfs_i[:, :])
    # a = ax.pcolormesh(np.linspace(0, 1, data.shape[1]), np.linspace(-45, 90, data.shape[0]),
    #                   data, shading='gouraud', linewidth=0, rasterized=True)
    # formatter = hpVis.ERBFormatter(100, 20000, unit='', places=0)
    #
    # ax.xaxis.set_major_formatter(formatter)
    # plt.colorbar(a)
    # ax = fig.add_subplot(1, 2, 2)
    # data = learned_map_bin_mean[:, :]
    # data = np.squeeze(hrtfs_c[:, :])
    # a = ax.pcolormesh(np.linspace(0, 1, data.shape[1]), np.linspace(-45, 90, data.shape[0]),
    #                   data, shading='gouraud', linewidth=0, rasterized=True)
    # formatter = hpVis.ERBFormatter(100, 20000, unit='', places=0)
    # ax.xaxis.set_major_formatter(formatter)
    # plt.colorbar(a)
    # plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
