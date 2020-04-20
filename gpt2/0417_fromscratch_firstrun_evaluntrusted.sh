for steps in 250000 500000;do
	bash mletrain_gpu_lr_trainsteps.sh 0 6.25e-4 $steps &  
	sleep 2m
	bash mletrain_gpu_lr_trainsteps.sh 1 6.25e-5 $steps &
	pid=$!
	wait $pid
	bash mletrain_gpu_lr_trainsteps.sh 0 1e-4 $steps &
	sleep 2m
	bash mletrain_gpu_lr_trainsteps.sh 1 1e-5 $steps &
	pid1=$!
	wait $pid1

done
