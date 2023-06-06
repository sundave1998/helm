#!bin/bash

set -e

# Test the checkpoints before sft under the order of training
# SUITE_NAME=test_convert
# PATH_MODEL_TO_EVAL=(
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0002000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0004000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0006000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0008000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0010000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0012000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0014000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0016000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0018000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0020000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0022000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0024000

#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0026000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0028000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0030000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0032000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0034000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0036000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0038000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0040000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0042000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0044000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0046000
#     /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/convert/megatronlm_to_huggingface/llama-7b/iter_0048000
# )

# test_sft
SUITE_NAME=test_sft
PATH_MODEL_TO_EVAL=(
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0002000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0004000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0006000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0008000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0010000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0012000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0014000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0016000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0018000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0020000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0022000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0024000

    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0026000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0028000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0030000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0032000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0034000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0036000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0038000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0040000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0042000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0044000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0046000
    /cpfs01/shared/Group-vpc-internal-ai-sys/checkpoints/sft/megatronllama_alpaca/7B/iter_0048000
)

WORK_DIR=/home/data/gaodawei.gdw/evaluation
BUILD_DIR=${WORK_DIR}/build
TARGET_CONF=${WORK_DIR}/run_specs_tiny.conf

# start
cd ${WORK_DIR}

# run evaluation
export PYTHONPATH=$PYTHONPATH:/home/data/gaodawei.gdw/helm/src/

# prepare run_specs.conf
if [ ! -d "${BUILD_DIR}" ]; then
  mkdir ${BUILD_DIR}
fi

# delete existing conf files
rm -rf ${BUILD_DIR}/*

for path_model in ${PATH_MODEL_TO_EVAL[*]}; 
do
  # clear the cache
  rm -rf ${WORK_DIR}/prod_env

	new_conf=${BUILD_DIR}/${path_model//'/'/'_'}.conf
	# create new run_specs_tiny.conf
	cp ${TARGET_CONF} ${new_conf}
	# replace by the specific model
	replace=s/MODEL/${path_model//'/'/'\/'}/g
	sed -i ${replace} ${new_conf}

  echo ${new_conf} ${path_model}

  /usr/bin/python /home/data/gaodawei.gdw/helm/src/helm/benchmark/run.py \
    --conf-paths ${new_conf} \
    -n 1 \
    --local \
    --suite ${SUITE_NAME} \
    --max-eval-instances 100 \
    --skip-completed-runs \
    --enable-local-huggingface-models ${path_model}

done


