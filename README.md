# astrokat-helper
Helper respository for astrokat. Includes tools for analysing the astrokat output.

# Install
First install `astrokat`. I.e. clone the repo, navigate there, and type `pip install .`, then go to this repo's root 
directory and repeat, `pip install .` . The python interpreter should then be able to execute all scripts found in here,
and a new command `when_is_patch_observable` is installed.

# Examples
## new observation workflow
- prepare new yaml file
- run `astrokat-observe` on it. It will likely fail and tell you the LST window
- Use `astrokat-lst` to translate that LST to UTC. Adapt the `yaml` accordingly.
- Run again. It should now work...
- Also check yaml file using `scripts/yaml_checker.py`

## targets
python ../astrokat/scripts/astrokat-targets.py --target scan_azel_with_nd_trigger '00:52:00.0' '00:00:00.0' --cat-path ../astrokat/catalogues

The script `scripts/corners_converter_angle_hourangle.py` helps to convert degree RA/DEC coordinates into the format required by astrokat.

## patch observability
Define a patch of the sky with right ascension and declination min/max values. You can display the dates when this
patch will be drift-scannable with the script

`when_is_patch_observable`

The command line options can be displayed with `when_is_patch_observable --help`, and example usage
is
`when_is_patch_observable --corners 6 20 -5 5 --date 2023-12-24 --buffer 30 --min_elevation 35 --max_elevation 50`