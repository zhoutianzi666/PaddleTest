model_item=gpt_stage2
dp_degree=1
mp_degree=1
pp_degree=1
bs_item=128
fp_item=fp16
run_mode=DP1-Sharding16
device_num=N2C16

model=gpt
micro_bs=8

cd ./benchmarks
bash ./test_tipc/gpt/dygraph/sharding/benchmark_common/prepare.sh
# run
bash ./test_tipc/gpt/dygraph/sharding/benchmark_common/run_benchmark.sh ${model_item} ${fp_item} ${dp_degree} ${mp_degree} ${pp_degree} ${micro_bs} ${bs_item} ${run_mode} ${device_num} 2>&1;
