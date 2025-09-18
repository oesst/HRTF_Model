import click
import logging
from pathlib import Path
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

import numpy as np
import soundfile as sf
from scipy import io
import scipy.signal as sp
from src.features import gtgram
import sounddevice as sd
from src.features import helpers_vis as hp_vis
import pandas as pd


import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
# set the path to the sound files
SOUND_FILES = ROOT / "data/raw/sound_samples/"
# create a list of the sound files
SOUND_FILES = np.sort(list(SOUND_FILES.glob("**/*.wav")))

# Define up to which frequency the data should be generated

print(SOUND_FILES)

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
    # Paths for datasets without HRTF filtering
    str_r_no_hrtf = (
        "data/processed_"
        + str(max_freq)
        + "Hz_no_HRTF_filtering/binaural_right_0_gammatone_"
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
    str_l_no_hrtf = (
        "data/processed_"
        + str(max_freq)
        + "Hz_no_HRTF_filtering/binaural_left_0_gammatone_"
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

    # Paths for datasets with HRTF filtering
    if diff_noise:
        str_r_hrtf = (
            "data/processed_"
            + str(max_freq)
            + "Hz_noise/binaural_right_0_gammatone_"
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
        str_l_hrtf = (
            "data/processed_"
            + str(max_freq)
            + "Hz_noise/binaural_left_0_gammatone_"
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
        str_r_hrtf = (
            "data/processed_"
            + str(max_freq)
            + "Hz/binaural_right_0_gammatone_"
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
        str_l_hrtf = (
            "data/processed_"
            + str(max_freq)
            + "Hz/binaural_left_0_gammatone_"
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

    # Paths for saving data
    path_data_r_no_hrtf = ROOT / str_r_no_hrtf
    path_data_l_no_hrtf = ROOT / str_l_no_hrtf
    path_data_r_hrtf = ROOT / str_r_hrtf
    path_data_l_hrtf = ROOT / str_l_hrtf

    # Check and create datasets without HRTF filtering
    if not path_data_r_no_hrtf.is_file() or not path_data_l_no_hrtf.is_file():
        path_data_r_no_hrtf.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Creating dataset without HRTF filtering: " + str_l_no_hrtf)

        data_r_no_hrtf = []
        data_l_no_hrtf = []

        for i in range(len(SOUND_FILES)):
            logging.info("Creating dataset for sound: " + SOUND_FILES[i].name)
            signal = sf.read(SOUND_FILES[i].as_posix())[0]
            signal_elevs = (1 - snr) * signal + snr * np.random.normal(0, np.abs(signal).max(), signal.shape[0])
            signal_elevs_c = (1 - snr) * signal + snr * np.random.normal(0, np.abs(signal).max(), signal.shape[0])

            twin, thop, fmin, fs = time_window, time_window / 2, 100, 44100

            y = gtgram.gtgram(signal_elevs, fs, twin, thop, freq_bands, fmin, max_freq)
            psd_i = 20 * np.log10(np.mean(y, axis=1) + np.finfo(np.float32).eps)

            y = gtgram.gtgram(signal_elevs_c, fs, twin, thop, freq_bands, fmin, max_freq)
            psd_c = 20 * np.log10(np.mean(y, axis=1) + np.finfo(np.float32).eps)

            for elev in range(25):  # Assuming 25 elevations
                data_l_no_hrtf.append({
                    "Sound": SOUND_FILES[i].name,
                    "Elevation": elev,
                    "Frequency_Bands": list(psd_i)
                })
                data_r_no_hrtf.append({
                    "Sound": SOUND_FILES[i].name,
                    "Elevation": elev,
                    "Frequency_Bands": list(psd_c)
                })

        df_l_no_hrtf = pd.DataFrame(data_l_no_hrtf)
        df_r_no_hrtf = pd.DataFrame(data_r_no_hrtf)
        print(df_l_no_hrtf)
        df_l_no_hrtf.to_pickle(path_data_l_no_hrtf.absolute())
        df_r_no_hrtf.to_pickle(path_data_r_no_hrtf.absolute())

    # Check and create datasets with HRTF filtering
    if not path_data_r_hrtf.is_file() or not path_data_l_hrtf.is_file():
        path_data_r_hrtf.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Creating dataset with HRTF filtering: " + str_l_hrtf)

        data_r_hrtf = []
        data_l_hrtf = []

        # read the HRIR data
        hrtf_path = (ROOT / "data/raw/hrtfs/hrir_{0:03d}.mat".format(participant_number)).resolve()
        # hrtf_path.mkdir(parents=False, exist_ok=True)
        hrir_mat = io.loadmat(hrtf_path.as_posix())

        # get the data for the left ear
        hrir_l = hrir_mat["hrir_l"]
        # get the data for the right ear
        hrir_r = hrir_mat["hrir_r"]
        # use always all elevations -> 50
        psd_all_i = np.zeros((len(SOUND_FILES), 25, freq_bands))
        psd_all_c = np.zeros((len(SOUND_FILES), 25, freq_bands))
        signal_all = np.zeros((len(SOUND_FILES), 25, freq_bands))

        formatter = hp_vis.ERBFormatter(0, 44100/2, unit='Hz', places=0)

        for i in range(0, psd_all_i.shape[0]):
            logging.info("Creating dataset for sound: " + SOUND_FILES[i].name)
            for i_elevs in range(psd_all_i.shape[1]):
                # load a sound sample
                signal = sf.read(SOUND_FILES[i].as_posix())[0]

                # read the hrir for a specific location
                hrir_elevs = np.squeeze(hrir_l[azimuth, i_elevs, :])
                # filter the signal
                signal_elevs = sp.filtfilt(hrir_elevs, 1, signal)
                # add noise to the signal
                std = np.abs(signal_elevs).max()

                # this should be the correct way to add noise
                signal_elevs = (1 - snr) * signal_elevs + snr * np.random.normal(0, std, signal_elevs.shape[0])

                # fig, axes = plt.subplots(3, 1)
                # fig.suptitle(f"Participant: {participant_number} Sound: {SOUND_FILES[i].name}, Elevation: {i_elevs}")
                # ax = axes[0]
                # ax.plot(signal)

                # ax = axes[1]
                # ax.specgram(signal, NFFT=256, Fs=44100, noverlap=128)

                # ax = axes[2]
                # ax.specgram(signal_elevs, NFFT=256, Fs=44100, noverlap=128)

                # ##### Sound Playback #####
                # signal_play = signal_elevs * (2**15 - 1) / np.max(np.abs(signal_elevs))
                # signal_play = signal_play.astype(np.int16)

                # # Start playback
                # sd.play(signal_play, 44100)

                # Wait for playback to finish before exiting

                # read the hrir for a specific location
                hrir_elevs = np.squeeze(hrir_r[azimuth, i_elevs, :])
                # filter the signal
                signal_elevs_c = sp.filtfilt(hrir_elevs, 1, signal)
                # add noise to the signal
                std = np.abs(signal_elevs_c).max()
                # add noise to the signal
                std = np.abs(signal_elevs_c).max()
                signal_elevs_c = (1 - snr) * signal_elevs_c + (snr * np.random.normal(0, std, signal_elevs_c.shape[0]))

                # Default gammatone-based spectrogram parameters
                time_window = 0.1
                twin = time_window
                thop = twin / 2
                fmin = 100
                fs = 44100

                y = gtgram.gtgram(signal, fs, twin, thop, freq_bands, fmin, max_freq)

                y = np.mean(y, axis=1)
                y = 20 * np.log10(y + np.finfo(np.float32).eps)
                signal_all[i, i_elevs, :] = y

                ###### Apply Gammatone Filter Bank ##############
                # ipsi side
                y = gtgram.gtgram(signal_elevs, fs, twin, thop, freq_bands, fmin, max_freq)

                y = np.mean(y, axis=1)
                y = 20 * np.log10(y + np.finfo(np.float32).eps)
                # add noise here
                # psd_all_i[i, i_elevs, :] = ((1 - snr) * y + snr * np.random.normal(0, std_noise, y.shape[0]))
                psd_all_i[i, i_elevs, :] = y
                # contralateral side
                y = gtgram.gtgram(signal_elevs_c, fs, twin, thop, freq_bands, fmin, max_freq)

                # fig, ax = plt.subplots(1, 1)

                # ax.yaxis.set_major_formatter(formatter)

                # Z = np.flipud(20 * np.log10(y))

                # img = ax.imshow(Z, extent=[0, len(signal)/fs, 1, 0], aspect='auto')
                # # ax.imshow(y, aspect="auto")
                # plt.show()

                y = np.mean(y, axis=1)
                y = 20 * np.log10(y + np.finfo(np.float32).eps)
                # psd_all_c[i, i_elevs, :] = ((1 - snr) * y + snr * np.random.normal(0, std_noise, y.shape[0]))
                psd_all_c[i, i_elevs, :] = y
                #################################################
            # fig, axes = plt.subplots(3, 1)
            # fig.suptitle(f"Participant: {participant_number} Sound: {SOUND_FILES[i].name}, Elevation: {i_elevs}")
            # ax = axes[0]
            # ax.xaxis.set_major_formatter(formatter)

            # ax.imshow(signal_all[i, :, :], extent=[0, 1, 0, freq_bands], aspect="auto")
            # ax.set_title("Original Sound")
            # ax.set_xlabel("Frequency Bands")
            # ax.set_ylabel("Elevation")
            # # set correct frequency bandas as xtick labels
            # print(formatter)

            # ax = axes[1]
            # ax.imshow(psd_all_i[i, :, :], aspect="auto")
            # ax.set_title("Left Ear")
            # ax.set_ylabel("Elevation")
            # ax.set_xlabel("Frequency Bands")

            # ax = axes[2]
            # ax.imshow(psd_all_c[i, :, :], aspect="auto")
            # ax.set_title("Right Ear")
            # ax.set_ylabel("Elevation")
            # ax.set_xlabel("Frequency Bands")

            # plt.show()

            for elev in range(25):  # Assuming 25 elevations
                data_l_hrtf.append({
                    "Sound": SOUND_FILES[i].name,
                    "Elevation": elev,
                    "Frequency_Bands": list(psd_all_i[i, elev, :])
                })
                data_r_hrtf.append({
                    "Sound": SOUND_FILES[i].name,
                    "Elevation": elev,
                    "Frequency_Bands": list(psd_all_c[i, elev, :])
                })

        df_l_hrtf = pd.DataFrame(data_l_hrtf)
        df_r_hrtf = pd.DataFrame(data_r_hrtf)
        print(df_l_hrtf)
        df_l_hrtf.to_pickle(path_data_l_hrtf.absolute())
        df_r_hrtf.to_pickle(path_data_r_hrtf.absolute())

        # np.save(path_data_r_hrtf.absolute(), psd_all_c)
        # np.save(path_data_l_hrtf.absolute(), psd_all_i)

    return (
        np.load(path_data_r_no_hrtf),
        np.load(path_data_l_no_hrtf),
        np.load(path_data_r_hrtf),
        np.load(path_data_l_hrtf),
    )


def main():
    """This script creates HRTF and non-HRTF filtered sound samples."""
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    normalize = False
    time_window = 0.1
    snrs = np.arange(0, 1.1, 0.1)
    snrs = np.array([0.0])
    freq_bandss = np.array([24, 128])
    azimuths = np.array([12])
    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11, 12, 15, 17, 18, 19, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147, 148, 152, 153, 154, 155, 156, 158, 162, 163, 165])
    max_freqs = np.array([20001])

    for _, participant_number in enumerate(participant_numbers):
        for _, snr in enumerate(snrs):
            for _, freq_bands in enumerate(freq_bandss):
                for _, azimuth in enumerate(azimuths):
                    for _, max_freq in enumerate(max_freqs):
                        create_data(
                            freq_bands, participant_number, snr, normalize, azimuth, time_window, max_freq=max_freq
                        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
