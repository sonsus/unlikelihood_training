gpu=$1
lr=$2
trainsteps=$3



CUDA_VISIBLE_DEVICES=$gpu nohup python run_gpt2.py      \
        --data-path /workhere/nluguidednlg/data/americanlit/     \
        --output-dir /workhere/fairseq/fairseq/checkpoint/gpt2/amerlit-train/     \
        --eval-split valid     \
        --train-n-steps 20000     \
        --validate-every 1000     \
        --sequence-tune-rate 0.0     \
        --mode train \
        --model-name from_scratch \
        --learning-rate $lr --train-n-steps $trainsteps --warmup_steps $(( $trainsteps / 10 )) \
	--batch-size 32 --seqlen 80 --gradient-accumulation-steps 2 > 0417_gpt2mle_scratch_${lr}_${trainsteps}.out &
pid=$!
wait $pid
