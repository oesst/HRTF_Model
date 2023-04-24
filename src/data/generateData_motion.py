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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
# set the path to the sound files
SOUND_FILES = ROOT / "data/raw/sound_samples/"
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob("**/*.wav"))

# Define up to which frequency the data should be generated


def create_data(
    freq_bands=24,
    participant_number=19,
    snr=0.2,
    normalize=False,
    azimuth=12,
    time_window=0.1,
    max_freq=20000,
    diff_noise=False,
):

    if diff_noise:
        str_r = (
            "data/processed_"
            + str(max_freq)
            + "Hz_noise_motion/binaural_right_0_gammatone_"
            + str(time_window)
            + "_window_{0:03d}".format(participant_number)
            + "_cipic_"
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
            + "Hz_noise_motion/binaural_left_0_gammatone_"
            + str(time_window)
            + "_window_{0:03d}".format(participant_number)
            + "_cipic_"
            + str(int(snr * 100))
            + "_srn_"
            + str(freq_bands)
            + "_channels_"
            + str((azimuth - 12) * 10)
            + "_azi_"
            + str(normalize)
            + "_norm.npy"
        )
    else:
        str_r = (
            "data/processed_"
            + str(max_freq)
            + "Hz_motion/binaural_right_0_gammatone_"
            + str(time_window)
            + "_window_{0:03d}".format(participant_number)
            + "_cipic_"
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
            + "Hz_motion/binaural_left_0_gammatone_"
            + str(time_window)
            + "_window_{0:03d}".format(participant_number)
            + "_cipic_"
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
        # read the HRIR data
        hrtf_path = (ROOT / "data/raw/hrtfs/hrir_{0:03d}.mat".format(participant_number)).resolve()
        # hrtf_path.mkdir(parents=False, exist_ok=True)
        hrir_mat = io.loadmat(hrtf_path.as_posix())

        # get the data for the left ear
        hrir_l = hrir_mat["hrir_l"]
        # get the data for the right ear
        hrir_r = hrir_mat["hrir_r"]



        # hrir_l_elevs_new = hrir_l[azimuth, :, :]
        # hrir_r_elevs_new = hrir_r[azimuth, :, :]
        
        # interpolate HRTFs between two measuring points in order to get a finer resultion
        resolution = 4

        hrir_l_elevs_new = np.zeros((resolution * 25, hrir_r.shape[2]))
        hrir_r_elevs_new = np.zeros((resolution * 25, hrir_r.shape[2]))
        for i in range(0, hrir_r.shape[2]):
            hrir_elevs = np.squeeze(hrir_l[azimuth, 0:25, i])
            x = np.linspace(0, 1, 25)
            # print(x.shape)
            f_interp = interp1d(x, hrir_elevs)
            x = np.linspace(0, 1, 25 * resolution)
            hrir_l_elevs_new[:, i] = f_interp(x)

            hrir_elevs = np.squeeze(hrir_r[azimuth, 0:25, i])
            x = np.linspace(0, 1, 25)
            # print(x.shape)
            f_interp = interp1d(x, hrir_elevs)
            x = np.linspace(0, 1, 25 * resolution)
            hrir_r_elevs_new[:, i] = f_interp(x)


        psd_all_i = np.zeros((len(SOUND_FILES), 25 * resolution, freq_bands))
        psd_all_c = np.zeros((len(SOUND_FILES), 25 * resolution, freq_bands))
        for i in range(0, psd_all_i.shape[0]):
            logging.info("Creating dataset for sound: " + SOUND_FILES[i].name)
            for i_elevs in range(psd_all_i.shape[1]):
                # load a sound sample
                signal = sf.read(SOUND_FILES[i].as_posix())[0]

                # read the hrir for a specific location
                hrir_elevs = np.squeeze(hrir_l_elevs_new[i_elevs, :])
                # filter the signal
                signal_elevs = sp.filtfilt(hrir_elevs, 1, signal)
                # add noise to the signal
                signal_elevs = (1 - snr) * signal_elevs + snr * np.random.random(signal_elevs.shape[0]) * signal.max()

                ##### Sound Playback #####
                # signal_play = signal_elevs * (2**15 - 1) / np.max(np.abs(signal_elevs))
                # signal_play = signal_play.astype(np.int16)
                #
                # # Start playback
                # play_obj = sa.play_buffer(signal_play, 1, 2, 44100)
                #
                # # Wait for playback to finish before exiting
                # play_obj.wait_done()

                # read the hrir for a specific location
                hrir_elevs = np.squeeze(hrir_r_elevs_new[i_elevs, :])
                # filter the signal
                signal_elevs_c = sp.filtfilt(hrir_elevs, 1, signal)
                # add noise to the signal
                signal_elevs_c = (1 - snr) * signal_elevs_c + snr * np.random.random(
                    signal_elevs_c.shape[0]
                ) * signal.max()

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
                psd_all_i[i, i_elevs, :] = y
                # contralateral side
                y = gtgram.gtgram(signal_elevs_c, fs, twin, thop, freq_bands, fmin, max_freq)
                y = np.mean(y, axis=1)
                y = 20 * np.log10(y + np.finfo(np.float32).eps)
                psd_all_c[i, i_elevs, :] = y
                #################################################

            signal_ones = np.ones_like(signal)

            elev_dims = hrir_r_elevs_new.shape[0]
            hrtf_r = np.zeros((elev_dims, 24))
            hrtf_l = np.zeros((elev_dims, 24))
            for i in range(elev_dims):
                signal_elevs = sp.filtfilt(hrir_r_elevs_new[i, :].squeeze(), 1, signal_ones)

                y = gtgram.gtgram(signal_elevs, fs, twin, thop, freq_bands, fmin, max_freq)
                y = np.mean(y, axis=1)
                y = 20 * np.log10(y + np.finfo(np.float32).eps)
                hrtf_r[i, :] = y.squeeze()

                signal_elevs = sp.filtfilt(hrir_l_elevs_new[i, :].squeeze(), 1, signal_ones)

                y = gtgram.gtgram(signal_elevs, fs, twin, thop, freq_bands, fmin, max_freq)
                y = np.mean(y, axis=1)
                y = 20 * np.log10(y + np.finfo(np.float32).eps)
                hrtf_l[i, :] = y.squeeze()
            # ax.pcolor(psd_motion_all_monaural[0, motion_spread:-motion_spread, :].squeeze())
            # print("hrtf:", hrtf.shape)

            # fig, axes = plt.subplots(2, 3, squeeze=False)
            # ax = axes[0, 0]
            # ax.set_title("HRTF left")
            # ax.pcolor(hrtf_l.squeeze())
            # ax = axes[0, 1]
            # ax.set_title("HRTF right")
            # ax.pcolor(hrtf_r.squeeze())
            # ax = axes[0, 2]
            # ax.set_title("Spectrogram of Sound")
            # hrir_elevs = np.ones_like(hrir_elevs)
            # signal_elevs = sp.filtfilt(hrir_elevs, 1, signal)

            # y = gtgram.gtgram(signal_elevs, fs, twin, thop, freq_bands, fmin, max_freq)
            # y = np.mean(y, axis=1)
            # y = 20 * np.log10(y + np.finfo(np.float32).eps)
            # print(y.shape)
            # ax.pcolor(np.tile(y,(25,1)))

            # ax = axes[1, 0]
            # ax.set_title("Filtered Sound")
            # # ax.pcolor(psd_motion_all_monaural[0, motion_spread:-motion_spread, :].squeeze())
            # ax.pcolor(psd_all_c[0, :, :].squeeze())
            # ax = axes[1, 1]
            # ax.pcolor(psd_all_i[0, :, :].squeeze())

            # ax = axes[1, 2]
            # ax.pcolor(psd_all_i[0, :, :].squeeze() / psd_all_c[0, :, :].squeeze())
            # plt.show()

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
    snrs = np.arange(0, 1.0, 0.1)  # Signal to noise ratio
    # snrs = np.array([0.2])  # Signal to noise ratio
    # snrs = np.array([0.0])  # Signal to noise ratio

    # freq_bandss = np.array([32, 64, 128]) # Frequency bands in resulting data
    freq_bandss = np.array([24])  # Frequency bands in resulting data
    # azimuths = np.arange(0, 25, 1)  # which azimuths to create
    azimuths = np.array([12])  # which azimuths to create
    participant_numbers = np.array(
        [
            1,
            2,
            3,
            8,
            9,
            10,
            11,
            12,
            15,
            17,
            18,
            19,
            20,
            21,
            27,
            28,
            33,
            40,
            44,
            48,
            50,
            51,
            58,
            59,
            60,
            61,
            65,
            119,
            124,
            126,
            127,
            131,
            133,
            134,
            135,
            137,
            147,
            148,
            152,
            153,
            154,
            155,
            156,
            158,
            162,
            163,
            165,
        ]
    )

    # define max frequency for gammatone filter bank
    max_freqs = np.array([20000])

    participant_numbers = participant_numbers[::-1]
    # snrs = snrs[::-1]
    # freq_bandss = freq_bandss[::-1]

    ########################################################################
    ########################################################################

    # walk over all parameter combinations
    for _, participant_number in enumerate(participant_numbers):
        for _, snr in enumerate(snrs):
            for _, freq_bands in enumerate(freq_bandss):
                for _, azimuth in enumerate(azimuths):
                    for _, max_freq in enumerate(max_freqs):
                        psd_all_c, psd_all_i = create_data(
                            freq_bands, participant_number, snr, normalize, azimuth, time_window, max_freq=max_freq,    diff_noise=True,

                        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
