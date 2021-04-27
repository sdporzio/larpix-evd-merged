 #!/bin/bash
#############################################################################
SCRIPT_FILE='./module0_evd.py'
GEOMETRY_FILE='./multi_tile_layout-2.1.16.yaml'
EVD_FILE='./raw_2021_04_02_22_38_12_CEST_evd.h5'
LIGHT_FILE='/home/dporzio/Data/Light/rwf_20210402_223712.data.root'
#############################################################################
# PROCESSING A SINGLE FILE
# -----------------------------------------------------------------------
python ${SCRIPT_FILE} \
        --geometry_file ${GEOMETRY_FILE} \
        --filename ${EVD_FILE} \
        --LDSfilename ${LIGHT_FILE}
# -----------------------------------------------------------------------
# OR, IF IN DOUBT
# python ${SCRIPT_FILE} \
#       --help
