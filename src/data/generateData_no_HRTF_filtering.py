# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from os import listdir
from os.path import isfile, join

import numpy as np
import soundfile as sf
from scipy import io
import scipy.signal as sp
from src.features import gtgram

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
# set the path to the sound files
SOUND_FILES = ROOT / "data/raw/sound_samples/"
# create a list of the sound files
SOUND_FILES = np.sort(list(SOUND_FILES.glob("**/*.wav")))

# Define up to which frequency the data should be generated


def create_data(
    freq_bands=128,
    snr=0.0,
    normalize=False,
    azimuth=12,
    time_window=0.1,
    max_freq=20000,
    diff_noise=False,
):

    str_r = (
        "data/processed_"
        + str(max_freq)
        + "Hz_no_HRTF_filtering/binaural_right_0_gammatone_"
        + str(time_window)
        + "_window_"
        + str(int(snr * 100))
        + "_srn_"
        + str(freq_bands)
        + "_channels_"
        + str((azimuth - 12) * 10)
        + "_azi_"
        + str(normalize)
        + "_norm.npy"
    )
    str_l = (
        "data/processed_"
        + str(max_freq)
        + "Hz_no_HRTF_filtering/binaural_left_0_gammatone_"
        + str(time_window)
        + "_window_"
        + str(int(snr * 100))
        + "_srn_"
        + str(freq_bands)
        + "_channels_"
        + str((azimuth - 12) * 10)
        + "_azi_"
        + str(normalize)
        + "_norm.npy"
    )

    path_data_r = ROOT / str_r
    path_data_l = ROOT / str_l

    # check if we can load the data from a file
    if path_data_r.is_file() and path_data_l.is_file():
        logging.info("Data set found. Loading from file : " + str_r)
        logging.info(path_data_l)
        return np.load(path_data_r), np.load(path_data_l)
    else:
        logging.info("Creating data set : " + str_l)

        # use always all elevations -> 50
        psd_all_i = np.zeros((len(SOUND_FILES), 25, freq_bands))
        psd_all_c = np.zeros((len(SOUND_FILES), 25, freq_bands))
        for i in range(0, psd_all_i.shape[0]):
            logging.info("Creating dataset for sound: " + SOUND_FILES[i].name)
            signal = sf.read(SOUND_FILES[i].as_posix())[0]

            # filter the signal
            signal_elevs = signal

            # Default gammatone-based spectrogram parameters
            time_window = 0.1
            twin = time_window
            thop = twin / 2
            fmin = 100
            fs = 44100

            ###### Apply Gammatone Filter Bank ##############
            # ipsi side
            y = gtgram.gtgram(signal_elevs, fs, twin, thop, freq_bands, fmin, max_freq)

            y = np.mean(y, axis=1)
            y = 20 * np.log10(y + np.finfo(np.float32).eps)
            psd_all_i[i, :, :] = y
            psd_all_c[i, :, :] = y
            #################################################

        np.save(path_data_r.absolute(), psd_all_c)
        np.save(path_data_l.absolute(), psd_all_i)

        return psd_all_c, psd_all_i


def main():
    """This script creates HRTF filtered sound samples of the sounds given in the folder SOUND_FILES.
    This is done for each participant's HRTF specified in participant_numbers.
    ALL ELEVATIONS (50) are taken to filter the data.

    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    normalize = False  # paramter is not considered

    time_window = 0.1  # time window for spectrogram in sec

    # Parameter to test
    snrs = np.arange(0, 1.1, 0.1)  # Signal to noise ratio
    # snrs = np.array([0.2])  # Signal to noise ratio
    snrs = np.array([0.0])  # Signal to noise ratio
    # freq_bandss = np.array([32, 64, 128]) # Frequency bands in resulting data
    freq_bandss = np.array([24, 128])  # Frequency bands in resulting data
    # azimuths = np.arange(0, 25, 1)  # which azimuths to create
    azimuths = np.array([12])  # which azimuths to create

    # define max frequency for gammatone filter bank
    max_freqs = np.array([20000])

    # snrs = snrs[::-1]
    # freq_bandss = freq_bandss[::-1]

    ########################################################################
    ########################################################################

    # walk over all parameter combinations
    for _, snr in enumerate(snrs):
        for _, freq_bands in enumerate(freq_bandss):
            for _, azimuth in enumerate(azimuths):
                for _, max_freq in enumerate(max_freqs):
                    psd_all_c, psd_all_i = create_data(
                        freq_bands, snr, normalize, azimuth, time_window, max_freq=max_freq
                    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
