# astrokat-helper
Helper respository for astrokat. Includes tools for analysing the astrokat output.

# Install
First install `astrokat`. I.e. clone the repo, navigate there, and type `pip install .`, then go to this repo's root 
directory and repeat, `pip install .` . The python interpreter should then be able to execute all scripts found in here,
and a new command `when_is_patch_observable` is installed.

# Examples
## new observation workflow
- Treat rising and setting completely separately.
- Make a reasonable guess of the scan LST range. For example, usually the second corner to rise/set is relevant for a rising/setting scan to start or end.
- Make a reasonable guess of the time it takes to calibrate before and after the scan.
- Prepare a yaml file for the scan but without calibrators.
- Look for calibration targets with `astrokat-targets.py` , i.e. run it on the scan corners (put those into a csv file). If needed extend the catalogue. Make sure to include the desired tags like `gain` `pol` `bp` and date and horizon. 
- Select calibrators and add them to the yaml. The single-dish calibrators need to be observed on centre and shifted off-centre as described in the observation doc. The script `scripts/yaml_checker.py` runs a few checks on yaml files, it will also check that the single dish calibrator pointings are offset correctly.
- run `astrokat-observe` on the yaml for both ends of the LST range and check the output. Especially target elevation ranges are important. Fine tune the LST range and if needed, swap calibrators.
- Also double check yaml file using `scripts/yaml_checker.py`
- Once the yaml file is correct, create all desired mutations from it (e.g. shifted scan lines, with and without calibrators etc.)

## targets
`python astrokat-targets.py --infile ../../astrokat-helper/yaml/jan_2024/patch_1_targets.csv --horizon 35 --cat-path ../catalogues/ --cal-tags gain bp pol --datetime '2024-01-01 00:00'`

The script `scripts/corners_converter_angle_hourangle.py` helps to convert degree RA/DEC coordinates into the format required by astrokat.

## patch observability
Define a patch of the sky with right ascension and declination min/max values. You can display the dates when this
patch will be drift-scannable with the script

`when_is_patch_observable`

The command line options can be displayed with `when_is_patch_observable --help`, and example usage
is
`when_is_patch_observable --corners 6 20 -5 5 --date 2023-12-24 --buffer 30 --min_elevation 35 --max_elevation 50`