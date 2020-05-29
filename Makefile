.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = hrtf_model
PYTHON_INTERPRETER = python3



# Default model parameter

model_name = 'a' # needs to be set
exp_name= '' # needs to be set

azimuth = 12
snr = 0.2
freq_bands = 128
max_freq = 20000
normalize = False
time_window = 0.1  # time window in sec
elevations = 25
# filtering parameters
normalization_type = 'sum_1'
sigma_smoothing = 0
sigma_gauss_norm = 1
# use the mean subtracted map as the learned map
mean_subtracted_map = True
ear = 'contra'

# if the variable clean is set the model file is deleted and newly calculated

save_figs = False
save_figs = 'svg'
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/generateData.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

single_participant:
	test $(exp_name)
	test $(participant_number)

	# create model file
	$(PYTHON_INTERPRETER) src/models/single_participant_localization_exp/localize_sound.py --model_name='single_participant' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
	$(PYTHON_INTERPRETER) src/visualization/single_participant/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='single_participant' --exp_name=$(exp_name) --azimuth=$(azimuth) --participant_number=$(participant_number) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)

all_participants:
		test $(exp_name)

		# create model file
		$(PYTHON_INTERPRETER) src/models/all_participants_localization_exp/localize_sound.py --model_name='single_participant' --exp_name=$(exp_name) --azimuth=$(azimuth) --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm) $(clean)
		$(PYTHON_INTERPRETER) src/visualization/all_participants/visualize_default.py --save_figs=$(save_figs) --save_type=$(save_type) --model_name='single_participant' --exp_name=$(exp_name) --azimuth=$(azimuth)  --snr=$(snr) --freq_bands=$(freq_bands) --max_freq=$(max_freq) --elevations=$(elevations) --mean_subtracted_map=$(mean_subtracted_map) --ear=$(ear) --normalization_type=$(normalization_type) --sigma_smoothing=$(sigma_smoothing) --sigma_gauss_norm=$(sigma_gauss_norm)




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
