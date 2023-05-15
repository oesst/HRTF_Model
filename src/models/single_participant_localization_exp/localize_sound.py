# -*- coding: utf-8 -*-
from re import A
from time import monotonic_ns
import click
import logging
from pathlib import Path
from src.data import generateData_motion as generateData_motion
from src.data import generateData 
from src.features import helpers as hp
from src.features import helpers_vis as hpVis
import numpy as np
import pickle
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

SOUND_FILES = ROOT / "data/raw/sound_samples/"
# create a list of the sound files
SOUND_FILES = list(SOUND_FILES.glob("**/*.wav"))


# Define whether figures should be saved
@click.command()
@click.option("--model_name", default="single_participant", help="Defines the model name.")
@click.option("--exp_name", default="single_participant_default", help="Defines the experiment name")
@click.option("--azimuth", default=12, help="Azimuth for which localization is done. Default is 12")
@click.option("--participant_number", default=9, help="CIPIC participant number. Default is 9")
@click.option("--snr", default=0.2, help="Signal to noise ration to use. Default is 0.2")
@click.option("--freq_bands", default=24, help="Amount of frequency bands to use. Default is 128")
@click.option("--max_freq", default=20000, help="Max frequency to use. Default is 20000")
@click.option("--elevations", default=25, help="Number of elevations to use 0-n. Default is 25 which equals 0-90 deg")
@click.option("--mean_subtracted_map", default=True, help="Should the learned map be mean subtracted. Default is True")
@click.option("--ear", default="contra", help="Which ear should be used, contra or ipsi. Default is contra")
@click.option(
    "--normalization_type",
    default="none",
    help="Which normalization type should be used sum_1, l1, l2. Default is sum_1",
)
@click.option("--sigma_smoothing", default=0.0, help="Sigma for smoothing kernel. 0 is off. Default is 0.")
@click.option("--sigma_gauss_norm", default=0.0, help="Sigma for gauss normalization. 0 is off. Default is 1.")
@click.option("--clean", is_flag=True)
def main(
    model_name="single_participant",
    exp_name="single_participant_default",
    azimuth=12,
    participant_number=9,
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
    for a single participant.
    """
    logger = logging.getLogger(__name__)
    logger.info("Localizing sounds for a single participant")

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    normalize = False
    time_window = 0.1  # time window in sec
    elevations = 100
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
            participant_number,
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
    else:
        # create Path
        exp_path.mkdir(parents=True, exist_ok=True)
        # create or read the data
        psd_all_c, psd_all_i = generateData_motion.create_data(
            freq_bands, participant_number, snr, normalize, azimuth, time_window, max_freq=max_freq, diff_noise=False
        )

        # Take only given elevations

        #elevations_ = np.arange(0,100,4)
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

        # create or read the data
        psd_all_c, psd_all_i = generateData_motion.create_data(
            freq_bands, participant_number, snr, normalize, azimuth, time_window, max_freq=max_freq, diff_noise=True
        )
        # Take only given elevations
        psd_all_c = psd_all_c[:, elevations, :]
        psd_all_i = psd_all_i[:, elevations, :]

        # filter data and integrate it
        psd_mono, psd_mono_mean, psd_binaural, psd_binaural_mean = hp.process_inputs(
            psd_all_i, psd_all_c, ear, normalization_type, sigma_smoothing, sigma_gauss_norm
        )

        # localize the sounds and save the results
        x_mono, y_mono = hp.localize_sound(psd_mono, learned_map)

        x_mono_mean = np.zeros(x_mono.shape)
        y_mono_mean = np.zeros(y_mono.shape)

        motion_spread = 10
        psd_motion_all_monaural = np.zeros_like(psd_binaural)
        print("psd_all_c:" ,psd_all_c.shape)
        for i in range(0 + motion_spread, len(elevations) - motion_spread):
            psd_motion = psd_all_c[:, i, :] / np.mean(psd_all_c[:, i - motion_spread : i + motion_spread, :], axis=1)
            # psd_motion =  np.mean(psd_all_c[:, i - motion_spread : i + motion_spread, :], axis=1)
            if motion_spread == 0:
                psd_motion =  psd_all_c[:, i , :]
            print("psd_motion:" ,psd_motion.shape)
            psd_motion_all_monaural[:, i, :] = psd_motion
            psd_motion = psd_motion[:, None, :]
            # localize the sounds and save the results
            a, b = hp.localize_sound(psd_motion, learned_map)
            a[:, :, 1] = i

            x_mono_mean[:, i, :], y_mono_mean[:, i] = a.squeeze(), b.squeeze()

            # localize the sounds and save the results
        print(":" ,x_mono_mean.shape)

        x_bin = np.zeros(x_mono.shape)
        y_bin = np.zeros(y_mono.shape)
        psd_motion_all_binaural = np.zeros_like(psd_binaural)
        for i in range(0 + motion_spread, len(elevations) - motion_spread):

            psd_motion_c = psd_all_c[:, i, :] / np.mean(psd_all_c[:, i - motion_spread : i + motion_spread, :], axis=1)

            psd_motion_i = psd_all_i[:, i, :] / np.mean(psd_all_i[:, i - motion_spread : i + motion_spread, :], axis=1)

            if ear.find("contra") >= 0:
                psd_motion = psd_motion_c / psd_motion_i
            else:
                psd_motion = psd_motion_i / psd_motion_c

            # psd_motion = np.mean(
            #     psd_binaural[:, i - motion_spread : i + motion_spread, :], axis=1
            # )
            if motion_spread == 0:
                psd_motion = psd_binaural[:, i, :]
            psd_motion_all_binaural[:, i, :] = psd_motion
            psd_motion = psd_motion[:, None, :]
            # localize the sounds and save the results
            a, b = hp.localize_sound(psd_motion, learned_map)
            a[:, :, 1] = i
            x_bin[:, i, :], y_bin[:, i] = a.squeeze(), b.squeeze()
        print(":" ,psd_motion_all_monaural.shape)

        
        fig, axes = plt.subplots(1, 3, squeeze=False)
        ax = axes[0, 0]

        if motion_spread == 0:
            motion_spread =1
        sound = 1
        c = ax.pcolor(psd_motion_all_monaural[sound, motion_spread:-motion_spread, :].squeeze())
        plt.colorbar(c,ax=ax)
        ax = axes[0, 1]
        ax.pcolor(psd_motion_all_binaural[sound, motion_spread:-motion_spread, :].squeeze())
        plt.colorbar(c,ax=ax)
        ax = axes[0, 2]
        c = ax.pcolor(learned_map)
        plt.colorbar(c,ax=ax)
        plt.show()

        # localize the sounds and save the results
        x_bin_mean, y_bin_mean = hp.localize_sound(psd_binaural_mean, learned_map)

        with exp_file.open("wb") as f:
            logger.info("Creating model file")
            pickle.dump([x_mono, y_mono, x_mono_mean, y_mono_mean, x_bin, y_bin, x_bin_mean, y_bin_mean], f)

    # fig = plt.figure(figsize=(20, 5))
    # # plt.suptitle('Single Participant')
    # # Monoaural Data (Ipsilateral), No Mean Subtracted
    # ax = fig.add_subplot(1, 4, 1)
    # # hpVis.plot_localization_result(x_mono, y_mono, ax, SOUND_FILES, scale_values=True, linear_reg=True)
    # ax.pcolormesh(psd_binaural_mean.mean(0))
    # print(psd_binaural_mean)
    # ax.set_title('Monoaural')
    # # hpVis.set_axis(ax)
    # ax.set_ylabel('Estimated Elevation [deg]')
    # plt.show()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
