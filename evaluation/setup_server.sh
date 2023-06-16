# #!bin/bash

# set -e
 
# SUITE_NAME=test_nl2sql

# PATH_HELM=/home/data/gaodawei.gdw/helm
# PATH_WORKDIR=${PATH_HELM}/evaluation


# cd ${PATH_WORKDIR}

# export PYTHONPATH=$PYTHONPATH:${PATH_HELM}/src/

# /usr/bin/python ${PATH_HELM}/src/helm/benchmark/presentation/summarize.py \
#         --suite ${SUITE_NAME}

# /usr/bin/python ${PATH_HELM}/src/helm/benchmark/server.py -o ${PATH_WORKDIR}/benchmark_output

#!bin/bash
source activate crfm-helm

set -e
 
SUITE_NAME=test_run

PATH_HELM=/mnt/data/youbangsun/FederatedScope/helm
PATH_WORKDIR=${PATH_HELM}/../helm-eval


cd ${PATH_WORKDIR}

# export PYTHONPATH=$PYTHONPATH:${PATH_HELM}/src/

# /usr/bin/python ${PATH_HELM}/src/helm/benchmark/presentation/summarize.py \
python ${PATH_HELM}/src/helm/benchmark/presentation/summarize.py \
        --suite ${SUITE_NAME}

python ${PATH_HELM}/src/helm/benchmark/server.py -o ${PATH_WORKDIR}/benchmark_output
