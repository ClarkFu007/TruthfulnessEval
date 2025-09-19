python3 tfqa_mc_eval.py \
--device cuda:0 \
--model-name ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf \
--data-path ./dataset \
--output-path output-path.json > main0.txt;

python3 tfqa_mc_eval.py \
--model-name ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-2x8-hf \
--device cuda:0 \
--data-path ./dataset \
--output-path output-path.json >> main0.txt;

python3 tfqa_mc_eval.py \
--model-name ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf \
--device cuda:0 \
--early-exit-layers 16,18,20,22,24,26,28,30,32 \
--data-path ./dataset \
--output-path output-path.json >> main0.txt; 

python3 tfqa_mc_eval.py \
--model-name ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-2x8-hf \
--device cuda:0 \
--early-exit-layers 16,18,20,22,24,26,28,30,32 \
--data-path ./dataset \
--output-path output-path.json >> main0.txt;

python3 tfqa_eval.py \
--model-name ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf \
--device cuda:0 \
--data-path ./dataset \
--output-path ./tfqa_result;

python3 tfqa_eval.py \
--model-name ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-2x8-hf \
--device cuda:0 \
--data-path ./dataset \
--output-path ./tfqa_result;

python3 tfqa_eval.py \
--model-name ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf \
--device cuda:0 \
--early-exit-layers 16,18,20,22,24,26,28,30,32 \
--data-path ./dataset \
--output-path ./tfqa_result \
--num-gpus 1;

python3 tfqa_eval.py \
--model-name ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-2x8-hf \
--device cuda:0 \
--early-exit-layers 16,18,20,22,24,26,28,30,32 \
--data-path ./dataset \
--output-path ./tfqa_result \
--num-gpus 1;