import matplotlib.pyplot as plt
import matplotlib as mpl
import src.features.filters as filters
import src.features.helpers_vis as hp_vis
import src.features.helpers as hp
import logging
import pickle
from pathlib import Path
import numpy as np
import click


hp_vis.set_layout(15)


ROOT = Path(__file__).resolve().parents[3]
# ROOT = Path('.').resolve().parents[3]

SOUND_FILES = ROOT / "data/raw/sound_samples/"
# create a list of the sound files
SOUND_FILES = np.sort(list(SOUND_FILES.glob("**/*.wav")))

# Define whether figures should be saved
@click.command()
# @click.option('--save_figs', default=False, help='Save the figures.')
@click.option("--save_figs", type=click.BOOL, default=False, help="Save figures")
@click.option("--save_type", default="svg", help="Define the format figures are saved.")
@click.option("--model_name", default="elevation_spectra_maps", help="Defines the model name.")
@click.option("--exp_name", default="spectral_maps", help="Defines the experiment name")
@click.option("--azimuth", default=12, help="Azimuth for which localization is done. Default is 12")
@click.option("--participant_numbers", help="CIPIC participant number. Default is None", default="40")
@click.option("--snr", default=0.0, help="Signal to noise ration to use. Default is 0.2")
@click.option("--freq_bands", default=128, help="Amount of frequency bands to use. Default is 128")
@click.option("--max_freq", default=20000, help="Max frequency to use. Default is 20000")
@click.option("--elevations", default=25, help="Number of elevations to use 0-n. Default is 25 which equals 0-90 deg")
def main(
    save_figs=False,
    save_type="svg",
    model_name="elevation_spectra_maps",
    exp_name="spectral_maps",
    azimuth=12,
    participant_numbers=40,
    snr=0.0,
    freq_bands=128,
    max_freq=20000,
    elevations=25,
    clean=False,
):

    logger = logging.getLogger(__name__)
    logger.info("Plotting elevation spectra map for different sounds")

    ########################################################################
    ######################## Set parameters ################################
    ########################################################################
    normalize = False
    time_window = 0.1  # time window in sec

    # make sure save type is given
    if not save_type or len(save_type) == 0:
        save_type = "svg"

    # if participant_numbers is not given we use all of them
    if not participant_numbers:
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

        exp_name_str = hp.create_exp_name(
            [
                exp_name,
                time_window,
                int(snr * 100),
                freq_bands,
                max_freq,
                len(participant_numbers),
                (azimuth - 12) * 10,
                normalize,
                elevations,
            ]
        )
        exp_path = ROOT / "models" / model_name
        exp_file = exp_path / exp_name_str
    else:
        # participant_numbers are given. need to be cast to int array
        participant_numbers = np.array([int(i) for i in participant_numbers.split(",")])

        exp_name_str = hp.create_exp_name(
            [
                exp_name,
                time_window,
                int(snr * 100),
                freq_bands,
                max_freq,
                participant_numbers,
                (azimuth - 12) * 10,
                normalize,
                elevations,
            ]
        )
        exp_path = ROOT / "models" / model_name
        exp_file = exp_path / exp_name_str

    ########################################################################
    ########################################################################

    fig_size = (20, 20)

    formatter = hp_vis.ERBFormatter(20, max_freq, unit="Hz", places=0)

    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), "rb") as f:
            logger.info("Reading model data from file")
            [ipsi_maps, contra_maps, ipsi_maps_no_HRTF, contra_maps_no_HRTF] = pickle.load(f)

        for i_par, par in enumerate(participant_numbers):
            offset = 0
            all_sounds = np.squeeze(ipsi_maps[i_par, :, :, :])  # shape: sounds x elevations x freq bands

            # calculate the cross correlation of the different sounds from different elevations
            selected_elevations = [0, 24]
            cross_correlation = np.zeros((len(all_sounds)*len(selected_elevations), len(all_sounds)*len(selected_elevations))) * np.nan
            all_sound_names = []
            max_corr = 0
            i1_max, i2_max = -1, -1
            max_sound1, max_sound2 = None, None
            for i_sound, sound1 in enumerate(all_sounds):
                sound_name_1 = SOUND_FILES[i_sound+offset].name.split(".")[0]
                all_sound_names.append(sound_name_1)
                for i_sound2, sound2 in enumerate(all_sounds):
                    sound_name_2 = SOUND_FILES[i_sound2+offset].name.split(".")[0]

                    if i_sound == i_sound2:
                        continue
                    # for i, i_elevation in enumerate(selected_elevations):
                    #     for j, i_elevation2 in enumerate(selected_elevations):
                    #         cross_correlation[i_sound*len(selected_elevations)+i, i_sound2*len(selected_elevations)+j] = np.corrcoef(sound[i_elevation], sound2[i_elevation2])[0, 1]
                    # cross_correlation[i_sound*len(selected_elevations), i_sound2*len(selected_elevations)] = np.corrcoef(sound[selected_elevations[0]], sound2[selected_elevations[0]])[0, 1]
                    # cross_correlation[i_sound*len(selected_elevations)+1, i_sound2*len(selected_elevations)+1] = np.corrcoef(sound[selected_elevations[1]], sound2[selected_elevations[1]])[0, 1]
                    corr1 = np.corrcoef(sound1[selected_elevations[0]], sound1[selected_elevations[1]])[0, 1]
                    corr2 = np.corrcoef(sound2[selected_elevations[0]], sound2[selected_elevations[1]])[0, 1]
                    corr3 = np.corrcoef(sound1[selected_elevations[0]], sound2[selected_elevations[1]])[0, 1]
                    corr4 = np.corrcoef(sound1[selected_elevations[1]], sound2[selected_elevations[0]])[0, 1]

                    # only plot correlations larger than 0.8
                    corr1 = np.nan if corr1 < 0.8 else corr1
                    corr2 = np.nan if corr2 < 0.8 else corr2
                    corr3 = np.nan if corr3 < 0.8 else corr3
                    corr4 = np.nan if corr4 < 0.8 else corr4

                    cross_correlation[i_sound*len(selected_elevations), i_sound2*len(selected_elevations)] = corr1
                    cross_correlation[i_sound*len(selected_elevations)+1, i_sound2*len(selected_elevations)+1] = corr2
                    cross_correlation[i_sound*len(selected_elevations), i_sound2*len(selected_elevations)+1] = corr3
                    cross_correlation[i_sound*len(selected_elevations)+1, i_sound2*len(selected_elevations)] = corr4

                    max_corr_tmp = np.max([max_corr, corr3])

                    if max_corr_tmp > max_corr and sound_name_1.find("orchestra") < 0 and sound_name_2.find("dog") < 0 and sound_name_2.find("laughing") < 0:   #
                        # save highest correlation and the corresponding sounds
                        max_corr = max_corr_tmp
                        i1_max, i2_max = i_sound, i_sound2
                        max_sound1 = sound1
                        max_sound2 = sound2
                        spectrum1 = sound1[selected_elevations[0]]
                        spectrum2 = sound2[selected_elevations[1]]

            # Display the results
            fig1, ax = plt.subplots(1, 1, figsize=fig_size)
            c = ax.matshow(cross_correlation, cmap="viridis")  # , vmin=-1, vmax=1)
            ax.set_title("Cross Correlation of different sounds")
            ax.set_xlabel("Sound")
            ax.set_ylabel("Sound")
            # set the sound names as labels on y axis
            # Create labels for each sound and elevation combination
            labels = []
            for sound1 in all_sound_names:
                # for elev in selected_elevations:
                # labels.append(f"{sound} ({elev}°)")

                labels.append(f"{sound1}")

            # Set the tick positions and labels
            ax.set_yticks(np.arange(0, 2*len(labels), 2)+0.25)
            ax.set_yticklabels(labels)
            ax.set_xticks(np.arange(0, 2*len(labels), 2)+0.25)
            ax.set_xticklabels(labels, rotation=90)
            # remove the grid lines
            ax.grid(False)
            for i in range(0, 2*len(labels), 2):
                ax.hlines(i - 0.5, -0.5, 2*len(labels) - 0.5, colors="w", linestyles="-", linewidth=1)
                ax.vlines(i - 0.5, -0.5, 2*len(labels) - 0.5, colors="w", linestyles="-", linewidth=1)

            plt.colorbar(c)

            fig2, ax2 = plt.subplots(1, 1, figsize=fig_size)
            # plot the best two sounds
            sound_name1 = all_sound_names[i1_max]
            sound_name2 = all_sound_names[i2_max]
            # normalize the spectra for better visibility
            spectrum1 = spectrum1 - np.mean(spectrum1)
            spectrum2 = spectrum2 - np.mean(spectrum2)
            ax2.xaxis.set_major_formatter(formatter)

            ax2.plot(np.linspace(0, 1, len(spectrum1)), spectrum1, label=f"{sound_name1} (elev {selected_elevations[0]})")
            ax2.plot(np.linspace(0, 1, len(spectrum2)), spectrum2, label=f"{sound_name2} (elev {selected_elevations[1]})")
            ax2.set_title(f"Spectra of two most correlated sounds (corr={max_corr:.2f})")
            ax2.set_xlabel("Frequency band")
            ax2.set_ylabel("Normalized amplitude")
            ax2.legend()
            ax2.grid(True)

            # plot all elevation spectra for these two sounds
            fig3, ax3 = plt.subplots(1, 2, figsize=fig_size)
            gap_size = 0.25 * np.max([np.max(max_sound1), np.max(max_sound2)])
            color_map = mpl.colormaps["viridis"]
            colors = color_map(np.linspace(0, 1, elevations))
            ax3[0].xaxis.set_major_formatter(formatter)
            ax3[1].xaxis.set_major_formatter(formatter)

            for i in range(elevations):
                ax3[0].plot(np.linspace(0, 1, len(max_sound1[i].T)), max_sound1[i].T + gap_size * i, label=f"Elev {i}", color=colors[i])
                ax3[1].plot(np.linspace(0, 1, len(max_sound2[i].T)), max_sound2[i].T + gap_size * i, label=f"Elev {i}", color=colors[i])
            ax3[0].set_title(f"Spectra of {sound_name1}")
            ax3[1].set_title(f"Spectra of {sound_name2}")
            for i_a, a in enumerate(ax3):
                a.set_xlabel("Frequency band")
                a.set_ylabel("Normalized amplitude")
                # plot legend outside the plot
                if i_a != 0:
                    a.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            if save_figs:

                exp_name_str = hp.create_exp_name(
                    [
                        exp_name,
                        time_window,
                        int(snr * 100),
                        freq_bands,
                        max_freq,
                        participant_numbers,
                        (azimuth - 12) * 10,
                        normalize,
                    ]
                )
                fig_save_path = (
                    ROOT
                    / "reports"
                    / "figures"
                    / exp_name_str
                    / model_name
                    / ("participant_" + str(par))
                )

                if not fig_save_path.exists():
                    fig_save_path.mkdir(parents=True, exist_ok=True)

                path_final = (
                    fig_save_path
                    / (
                        exp_name
                        + "Cross_Correlation_"
                        + "_"
                        + "."
                        + save_type
                    )
                ).as_posix()
                fig1.savefig(path_final, dpi=300, transparent=True)
                logger.info("Writing File :" + path_final)

                path_final = (
                    fig_save_path
                    / (
                        exp_name
                        + "Highest_correlation_sounds"
                        + "."
                        + save_type
                    )
                ).as_posix()
                fig2.savefig(path_final, dpi=300, transparent=True)
                logger.info("Writing File :" + path_final)

                path_final = (
                    fig_save_path
                    / (
                        exp_name
                        + "All_elevation_spectra"
                        + "."
                        + save_type
                    )
                ).as_posix()
                fig3.savefig(path_final, dpi=300, transparent=True)
                logger.info("Writing File :" + path_final)
            else:
                plt.show()

    else:
        logger.error("No data set found. Run model first!")
        logger.error(exp_file)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
