 #!/bin/bash
#############################################################################
### MAIN SETTINGS
SCRIPT_FILE='./module0_evd.py' # Event display file
GEOMETRY_FILE='./multi_tile_layout-2.1.16.yaml' # Location of the geometry file
EVD_DIRECTORY='/data/LArPix/SingleModule_March2021/TPC12/dataRuns/evdData/good_michel' # Directory containing the charge (.h5) files
LIGHT_DIRECTORY='/data/LRS/Converted' # Directory containing the light (.root) files
EVD_FILE='michel_datalog_2021_04_05_17_28_34_CEST_evd.h5' # Charge file (.h5 format)
# No light file input needed, should be looked for automatically
#############################################################################
### OTHER SETTINGS
TRACK_OFFSET=100 # Vertical y-axis offset between track an hits for better visualization
#############################################################################
# PROCESSING A SINGLE FILE
# -----------------------------------------------------------------------
python ${SCRIPT_FILE} \
        --geometry_file ${GEOMETRY_FILE} \
        --filename ${EVD_DIRECTORY}/${EVD_FILE} \
        --LDSdirectory ${LIGHT_DIRECTORY} \
        --trackOffset ${TRACK_OFFSET}
# -----------------------------------------------------------------------
# OR, IF IN DOUBT
# python ${SCRIPT_FILE} \
#       --help
