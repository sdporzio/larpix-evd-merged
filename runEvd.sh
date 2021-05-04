 #!/bin/bash
#############################################################################
SCRIPT_FILE='./module0_evd.py'
GEOMETRY_FILE='./multi_tile_layout-2.1.16.yaml'
EVD_FILE='/home/dporzio/Data/MichelLimited/michel_datalog_2021_04_05_17_28_34_CEST_evd_limited.h5'
LIGHT_DIRECTORY='/home/dporzio/Data/MichelLimited'
#############################################################################
# PROCESSING A SINGLE FILE
# -----------------------------------------------------------------------
python ${SCRIPT_FILE} \
        --geometry_file ${GEOMETRY_FILE} \
        --filename ${EVD_FILE} \
        --LDSdirectory ${LIGHT_DIRECTORY}
# -----------------------------------------------------------------------
# OR, IF IN DOUBT
# python ${SCRIPT_FILE} \
#       --help
