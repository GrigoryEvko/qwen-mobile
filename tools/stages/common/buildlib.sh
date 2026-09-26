# shellcheck shell=bash
# The parts that each build recipe of a phone stage shares (tools/stages/*/build.sh).
# Source this file, do not execute it. It is valid on the host and in the Snapdragon container,
# because the container has the repository at /workspace and its current directory is /workspace:
#
#   source tools/stages/common/buildlib.sh
#
# A stage library must have the compiler flags of scripts/build-native.sh, or its bytes differ from
# the library of the app for a reason that no measurement shows. Thus a recipe takes the flags from
# the CMake preset of the app and adds only its own flags.

# Print one cache variable of the Snapdragon preset of a CMakeUserPresets.json.
#
#   preset_flag TREE FLAG
#
# TREE is the source tree that holds CMakeUserPresets.json, and FLAG is the name of the variable,
# for example CMAKE_C_FLAGS.
preset_flag() {
    python3 -c "
import json, sys
presets = json.load(open(sys.argv[1]))[\"configurePresets\"]
preset = [p for p in presets if p[\"name\"] == \"arm64-android-snapdragon\"][0]
print(preset[\"cacheVariables\"][sys.argv[2]])
" "$1/CMakeUserPresets.json" "$2"
}
