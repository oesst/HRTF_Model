#HRTF Model for Sound Source Localization

==============================

Computational model for sound source localization in the vertical plane using binaural integration based on learned HRTFs.


## Experiments

Experiments described here are found in _src/models/_ as separate folders.

* single_participant_exp : Experiments show localization results for a single participant (_localize_sound.py_) in the median plane. A variation of this experiment is the _localize_sound_differten_azi.py_ which allows to choose the azimuth angle from which sounds originate. The learned map remains at 0 &deg;.

* hrtfs_comparison_exp : Here we compare the resulted learned spectral map with the actual HRTF of a participant with calculated correlation coefficients.

* all_participants_exp: Experiments show localization results over all participants (_localize_sound.py_) in the median plane. A variation of this experiment is the _localize_sound_differten_azi.py_ which allows to choose the azimuth angle from which sounds originate. The learned map remains at 0 &deg;.

* ...

The HRTFs are taken from the CIPIC database  [V. R. Algazi, R. O. Duda, D. M. Thompson and C. Avendano, “The CIPIC HRTF Database,” Proc. 2001 IEEE Workshop on Applications of Signal Processing to Audio and Electroacoustics, pp. 99-102, Mohonk Mountain House, New Paltz, NY, Oct. 21-24, 2001.] (https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/)

The sound stimuli are taken from the natural sound stimuli set of Josh McDermott's group (see [stimuli](http://mcdermottlab.mit.edu/svnh/Natural-Sound/Stimuli.html)) and have been previously published in [Norman-Haignere et al., 2015, Neuron 88, 1281–1296 December 16, 2015 Elsevier Inc.](http://dx.doi.org/10.1016/j.neuron.2015.11.035) 

## Installing development requirements
------------

    pip install -r requirements.txt

==============================

Project Organization as from cookiecutter
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## References
<a id="1">[1]</a>
Algazi, V Ralph and Duda, Richard O and Thompson, Dennis M and Avendano, Carlos (2001).
The cipic hrtf database.
Proceedings of the 2001 IEEE Workshop on the Applications of Signal Processing to Audio and Acoustics (Cat. No. 01TH8575).
