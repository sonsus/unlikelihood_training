gpu=$1
lr=$2
traineps=$3
expdate=$4


CUDA_VISIBLE_DEVICES=$gpu nohup python run_gpt2.py      \
        --data-path ../jsonl_data/americanlit/     \
        --output-dir checkpoints/gpt2/0420_amerlit_scratch_eval_untrusted/     \
        --eval-split valid     \
        --validate-every 12500     \
        --sequence-tune-rate 0.0     \
        --mode train \
        --model-name from_scratch \
	--learning-rate $lr --num-train-epochs $traineps --warmup-steps $(( $(( $traineps / 10 )) * 12500 ))  \
	--batch-size 16 --seqlen 80 --gradient-accumulation-steps 4 > gpt2mle_scratch_${lr}_${traineps}_date${expdate}.out &
pid=$!
wait $pid
