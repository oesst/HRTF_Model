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
@click.option("--exp_name", default="test", help="Defines the experiment name")
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
    exp_name="unfiltered",
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
        print(participant_numbers)

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

    fig_size = (15, 9)
    # fig_size = (20, 14)

    formatter = hp_vis.ERBFormatter(20, max_freq, unit="Hz", places=0)

    separator = 4

    # check if model results exist already and load
    if exp_path.exists() and exp_file.is_file():
        # try to load the model files
        with open(exp_file.as_posix(), "rb") as f:
            logger.info("Reading model data from file")
            [ipsi_maps, contra_maps, ipsi_maps_no_HRTF, contra_maps_no_HRTF] = pickle.load(f)

        for i_par, par in enumerate(participant_numbers):

            for i_sound, sound in enumerate(SOUND_FILES):
                sound = sound.name.split(".")[0]

                # if not sound.find("crow") >= 0 and not sound.find("geese") >= 0:
                #     continue

                # IPSI
                fig, axes = plt.subplots(2, 4, figsize=(fig_size[0], fig_size[1]), squeeze=False, sharex=False, sharey=False)

                fig.suptitle("Participant: " + str(par) + " Sound: " + sound)

                cbar_size = 1

                # First row - mesh plots
                ax = axes[0, 0]
                ax.set_title("Sound (mesh)")
                ax.xaxis.set_major_formatter(formatter)
                g = ax.imshow(np.squeeze(ipsi_maps_no_HRTF[i_par, i_sound]), extent=[0, 1, 0, elevations], interpolation='bilinear', aspect='auto')
                # cbar = plt.colorbar(g, ax=ax, pad=0.01, shrink=cbar_size)
                # ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("Elevation")

                # Second row - line plots
                ax = axes[1, 0]
                # ax.set_title("Sound (lines)")
                ax.xaxis.set_major_formatter(formatter)
                for i in range(ipsi_maps_no_HRTF[i_par, i_sound].shape[0]):
                    data = ipsi_maps_no_HRTF[i_par, i_sound][i, :]
                    ax.plot(np.linspace(0, 1, len(data)), data+i*separator, label=str(i))
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("Energy [au]")

                # First row - Ipsi mesh
                ax = axes[0, 1]
                ax.set_title("Ipsi (mesh)")
                ax.xaxis.set_major_formatter(formatter)
                g = ax.imshow(np.squeeze(ipsi_maps[i_par, i_sound]), extent=[0, 1, 0, elevations], interpolation='bilinear', aspect='auto')
                # cbar = plt.colorbar(g, ax=ax, pad=0.01, shrink=cbar_size)
                # ax.set_xlabel("Frequency [Hz]")
                # ax.set_ylabel("Elevation")

                # Second row - Ipsi lines
                ax = axes[1, 1]
                # ax.set_title("Ipsi (lines)")
                ax.xaxis.set_major_formatter(formatter)
                for i in range(ipsi_maps[i_par, i_sound].shape[0]):
                    data = ipsi_maps[i_par, i_sound][i, :]
                    ax.plot(np.linspace(0, 1, len(data)), data+i*separator, label=str(i))
                ax.set_xlabel("Frequency [Hz]")
                # ax.set_ylabel("Energy [au]")

                # First row - Contra mesh
                ax = axes[0, 2]
                ax.set_title("Contra (mesh)")
                ax.xaxis.set_major_formatter(formatter)
                g = ax.imshow(np.squeeze(contra_maps[i_par, i_sound]), extent=[0, 1, 0, elevations], interpolation='bilinear', aspect='auto')
                # cbar = plt.colorbar(g, ax=ax, pad=0.01, shrink=cbar_size)
                # ax.set_xlabel("Frequency [Hz]")
                # ax.set_ylabel("Elevation")

                # Second row - Contra lines
                ax = axes[1, 2]
                # ax.set_title("Contra (lines)")
                ax.xaxis.set_major_formatter(formatter)
                for i in range(contra_maps[i_par, i_sound].shape[0]):
                    data = contra_maps[i_par, i_sound][i, :]
                    ax.plot(np.linspace(0, 1, len(data)), data+i*separator, label=str(i))
                ax.set_xlabel("Frequency [Hz]")
                # ax.set_ylabel("Energy [au]")

                # First row - Integrated mesh
                ax = axes[0, 3]
                ax.set_title("Integrated (mesh)")
                ax.xaxis.set_major_formatter(formatter)
                g = ax.imshow(np.squeeze(contra_maps[i_par, i_sound])/np.squeeze(ipsi_maps[i_par, i_sound]), extent=[0, 1, 0, elevations], interpolation='bilinear', aspect='auto')
                cbar = plt.colorbar(g, ax=ax, pad=0.01, shrink=cbar_size)
                cbar.set_label('Energy [au]')
                # ax.set_xlabel("Frequency [Hz]")
                # ax.set_ylabel("Elevation")

                # Second row - Integrated lines
                ax = axes[1, 3]
                # ax.set_title("Integrated (lines)")
                ax.xaxis.set_major_formatter(formatter)
                for i in range(contra_maps[i_par, i_sound].shape[0]):
                    data = (contra_maps[i_par, i_sound][i, :]/ipsi_maps[i_par, i_sound][i, :])
                    ax.plot(np.linspace(0, 1, len(data)), data+i*separator*0.01, label=str(i))

                ax.set_xlabel("Frequency [Hz]")
                # ax.set_ylabel("Energy [au]")

                # plt.tight_layout()

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
                            + "_raw_maps_ipsi_"
                            + str(sound)
                            + "_"
                            + str(elevations)
                            + "."
                            + save_type
                        )
                    ).as_posix()
                    plt.savefig(path_final, dpi=300, transparent=True)

                    logger.info("Writing File :" + path_final)
                    plt.close()
                else:
                    plt.show()

    else:
        logger.error("No data set found. Run model first!")
        logger.error(exp_file)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
