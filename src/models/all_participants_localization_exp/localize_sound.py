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

SOUND_FILES = ROOT / "data/raw/sound_samples/"
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob("**/*.wav"))

# Define whether figures should be saved
@click.command()
@click.option("--model_name", default="all_participants", help="Defines the model name.")
@click.option("--exp_name", default="localization_default", help="Defines the experiment name")
@click.option("--azimuth", default=12, help="Azimuth for which localization is done. Default is 12")
@click.option("--snr", default=0.2, help="Signal to noise ration to use. Default is 0.2")
@click.option("--freq_bands", default=128, help="Amount of frequency bands to use. Default is 128")
@click.option("--max_freq", default=20000, help="Max frequency to use. Default is 20000")
@click.option("--elevations", default=25, help="Number of elevations to use 0-n. Default is 25 which equals 0-90 deg")
@click.option("--mean_subtracted_map", default=True, help="Should the learned map be mean subtracted. Default is True")
@click.option("--ear", default="contra", help="Which ear should be used, contra or ipsi. Default is contra")
@click.option(
    "--normalization_type",
    default="sum_1",
    help="Which normalization type should be used sum_1, l1, l2. Default is sum_1",
)
@click.option("--sigma_smoothing", default=0, help="Sigma for smoothing kernel. 0 is off. Default is 0.")
@click.option("--sigma_gauss_norm", default=1, help="Sigma for gauss normalization. 0 is off. Default is 1.")
@click.option("--clean", is_flag=True)
def main(
    model_name="all_participants",
    exp_name="localization_default",
    azimuth=12,
    snr=0.2,
    freq_bands=24,
    max_freq=20000,
    elevations=25,
    mean_subtracted_map=True,
    ear="ipsi",
    normalization_type="sum_1",
    sigma_smoothing=0,
    sigma_gauss_norm=1,
    clean=False,
):
    """This script takes the filtered data and tries to localize sounds with a learned map
    for all participants.
    """
    logger = logging.getLogger(__name__)
    logger.info("Localizing sounds for all participants")

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    # participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11,
    #                                 12, 15, 17, 18, 19, 20, 21, 27, 28, 33, 40])

    participant_numbers = np.array([1, 2, 3, 8, 9, 10, 11, 12, 15, 17, 18, 19, 20, 21, 27, 28, 33, 40, 44, 48, 50, 51, 58, 59, 60, 61, 65, 119, 124, 126, 127, 131, 133, 134, 135, 137, 147, 148, 152, 153, 154, 155, 156, 158, 162, 163, 165])
    normalize = False
    time_window = 0.1  # time window in sec

    elevations = np.arange(0, elevations, 1)
    ########################################################################
    ########################################################################

    # create unique experiment name
    exp_name_str = hp.create_exp_name(
        [
            exp_name,
            normalization_type,
            sigma_smoothing,
            sigma_gauss_norm,
            mean_subtracted_map,
            time_window,
            int(snr * 100),
            freq_bands,
            max_freq,
            (azimuth - 12) * 10,
            normalize,
            len(elevations),
            ear,
        ]
    )

    exp_path = ROOT / "models" / model_name
    exp_file = exp_path / exp_name_str
    # check if model results exist already and load
    if not clean and exp_path.exists() and exp_file.is_file():

        # try to load the model files
        with exp_file.open("rb") as f:
            logger.info("Reading model data from file")
            [x_mono, y_mono, x_mono_mean, y_mono_mean, x_bin, y_bin, x_bin_mean, y_bin_mean] = pickle.load(f)
            print(x_mono.shape)
    else:
        x_mono = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations), 2))
        y_mono = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations)))
        x_mono_mean = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations), 2))
        y_mono_mean = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations)))
        x_bin = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations), 2))
        y_bin = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations)))
        x_bin_mean = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations), 2))
        y_bin_mean = np.zeros((len(participant_numbers), len(SOUND_FILES), len(elevations)))
        for i_par, par in enumerate(participant_numbers):

            # create or read the data
            psd_all_c, psd_all_i = generateData.create_data(
                freq_bands, par, snr, normalize, azimuth, time_window, max_freq=max_freq, diff_noise=False
            )

            # Take only given elevations
            psd_all_c = psd_all_c[:, elevations, :]
            psd_all_i = psd_all_i[:, elevations, :]

            # filter data and integrate it
            psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = hp.process_inputs(
                psd_all_i, psd_all_c, ear, normalization_type, sigma_smoothing, sigma_gauss_norm
            )

            # create map from defined processed data
            if mean_subtracted_map:
                learned_map = psd_binaural_mean.mean(0)
            else:
                learned_map = psd_binaural.mean(0)

            # create or read the data with different noise
            psd_all_c, psd_all_i = generateData.create_data(
                freq_bands, par, snr, normalize, azimuth, time_window, max_freq=max_freq, diff_noise=True
            )

            # Take only given elevations
            psd_all_c = psd_all_c[:, elevations, :]
            psd_all_i = psd_all_i[:, elevations, :]

            # filter data and integrate it
            psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = hp.process_inputs(
                psd_all_i, psd_all_c, ear, normalization_type, sigma_smoothing, sigma_gauss_norm
            )

            # localize the sounds and save the results
            x_mono[i_par, :, :, :], y_mono[i_par, :] = hp.localize_sound(psd_mono, learned_map)

            # localize the sounds and save the results
            x_mono_mean[i_par, :, :, :], y_mono_mean[i_par, :, :] = hp.localize_sound(psd_mono_mean, learned_map)

            # localize the sounds and save the results
            x_bin[i_par, :, :, :], y_bin[i_par, :, :] = hp.localize_sound(psd_binaural, learned_map)

            # localize the sounds and save the results
            x_bin_mean[i_par, :, :, :], y_bin_mean[i_par, :, :] = hp.localize_sound(psd_binaural_mean, learned_map)

        # create Path
        exp_path.mkdir(parents=True, exist_ok=True)
        with exp_file.open("wb") as f:
            logger.info("Creating model file")
            pickle.dump([x_mono, y_mono, x_mono_mean, y_mono_mean, x_bin, y_bin, x_bin_mean, y_bin_mean], f)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
