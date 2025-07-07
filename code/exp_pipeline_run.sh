GPU_ID=3
PROJECT_DIR="./"
# 输入bridge model path 获得改写数据
python get_rewrite_data_after_dpo.py \
    --model_dir ${PROJECT_DIR}/models/my_bridge_model/grpo_merge_lora_0706_bm25 \
    --data_dir ${PROJECT_DIR}/data/retrieval-test200 \
    --save_dir ${PROJECT_DIR}/data/retrieval-test200-grpo_0706_bm25 \
    --gpu_id ${GPU_ID} 

# 根据改写数据获得测评结果
rm -rf  ${PROJECT_DIR}/data/retrieval-test200-grpo_0706_bm25/summary_test_result.json
python eval_retriever.py \
    --model_path ${PROJECT_DIR}/models/ToolBench_IR_bert_based_uncased \
    --phase test \
    --train_type dpo \
    --retriever_type bm25 \
    --gpu_id ${GPU_ID} \
    --data_root  ${PROJECT_DIR}/data/retrieval-test200-grpo_0706_bm25