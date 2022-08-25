declare -a env=("HalfCheetah" "Hopper" "Walker2d" "Swimmer" "Ant" "Humanoid")

for index in "${!env[@]}"
do
    for ((i=0;i<5;i+=1))
    do

	python main.py --env-name "${env[$index]}-v3" --algo ppo --clip-param 0.2 --use-gae --log-dir "logs/${env[$index]}/PPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits

	python main.py --env-name "${env[$index]}-v3" --algo ppo --clip-param 0.2 --alpha 3e-7 --use-gae --log-dir "logs/${env[$index]}/SPPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --augment-type "shifted"

	python main.py --env-name "${env[$index]}-v3" --algo ppo --clip-param 0.2 --alpha 3e-7 --use-gae --log-dir "logs/${env[$index]}/IPPO-${i}" --seed $i --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --augment-type "invariant"
	
	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --log-dir "logs/${env[$index]}/TRPO-${i}" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --max-kl 0.01 --damping 1e-1 --l2-reg 1e-3
	
	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --alpha 3e-7 --log-dir "logs/${env[$index]}/STRPO-${i}" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --max-kl 0.01 --damping 1e-1 --l2-reg 1e-3 --augment-type "shifted"

	python main.py --env-name "${env[$index]}-v3" --algo trpo --use-gae --alpha 3e-7 --log-dir "logs/${env[$index]}/ITRPO-${i}" --seed $i --log-interval 1 --num-steps 4096 --num-processes 1 --ppo-epoch 1 --num-mini-batch 1 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --max-kl 0.01 --damping 1e-1 --l2-reg 1e-3 --augment-type "invariant"	     	
	
	python baselines.py --env-name "${env[$index]}-v3" --algo sac --seed $i --num-env-steps 1000000 --log-dir "logs/${env[$index]}/SAC-${i}"
	
	python baselines.py --env-name "${env[$index]}-v3" --algo td3 --seed $i --num-env-steps 1000000 --log-dir "logs/${env[$index]}/TD3-${i}"
	
    done
done
