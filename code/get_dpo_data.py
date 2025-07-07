# 针对每一个sft模型（G1, G2, G3）
# 推理bridge model, 获得4组结果
# 针对每一类检索器
# 利用4组结果进行检索，获得四组检索得分
# 将四组检索得分根据大小排序，取最高与最低即可获得DPO数据集


import json
import argparse
import pickle
import pandas as pd
from src.utils import initialize_retriever,read_query
from src.api_evaluator import APIEvaluator
from vllm import LLM, SamplingParams
import os
import tqdm



def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    retrieval_data_dir = args.retrieval_data_dir
    sft_data_dir = args.sft_data_dir
    dpo_data_dir = args.dpo_data_dir
    bridge_model_dir = args.bridge_model_dir
    retriever_model_path = args.retriever_model_path
    sub_data_name = args.sub_data_name
   
    # 针对每一个数据集和每一类检索器做实验 所以一共将得到 3*4=12类结果
    bridge_model_path = args.bridge_model_path
    if args.bridge_model_dir:    
        print(f"开始处理{sub_data_name}......")
        bridge_model_path = os.path.join(bridge_model_dir, sub_data_name)
    
    
    # 推理bridge model, 获得4组结果
    # 如果生成好了，则直接读取之
    multi_output_data_path = os.path.join(dpo_data_dir, "dpo_"+sub_data_name+"_wo_retrieval.json")
    results = None
    if os.path.exists(multi_output_data_path):
        results = json.load(open(multi_output_data_path,"r", encoding="utf-8"))
    else:
        # 否则则调用vllm进行bridge model 的推理
        # Define sampling parameters from the parsed arguments
        sampling_params = SamplingParams(temperature=args.temperature, n=args.n, max_tokens=args.max_tokens)
        # Initialize LLM with model path and parallelization setting
        llm = LLM(
            model=bridge_model_path,
            trust_remote_code=True,
            tokenizer=bridge_model_path,
            dtype='bfloat16',
            tokenizer_mode='auto',
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu,
            max_num_seqs=128
        )

        data_path = os.path.join(sft_data_dir, sub_data_name+"_trainset.json")
        

        # Read input data from JSON file
        prompts = []
        q_ids = []
        with open(data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        for data in test_data:
            instruction = data["instruction"]
            q_id = data["id"]
            q_ids.append(q_id)
            text = instruction
            if args.template == 'llama2':
                    prompts.append(f"<s> [INST] {text} [/INST]")
            elif args.template == 'llama3':
                prompts.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
            elif args.template == 'alpaca':
                prompts.append(f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{text}\n\n### Response:\n")
        print(len(prompts))
        outputs = llm.generate(prompts, sampling_params)
        # 获得推理结果，形成一个字典列表
        print(outputs)
        print(outputs[0])
        results = []
        for i in range(len(outputs)):
            if args.n > 1:
                result = {
                    "qid":test_data[i]["id"],
                    "instruction": test_data[i]["instruction"],
                    "gt_output": test_data[i]["output"],
                    "pred_output": [outputs[i].outputs[j].text.replace('\n', '') for j in range(len(outputs[i].outputs))]
                }
                results.append(result)
            elif args.n == 1:
                result = {
                    "qid":test_data[i]["id"],
                    "instruction": test_data[i]["instruction"],
                    "gt_output": test_data[i]["output"],
                    "pred_output": outputs[i].outputs[0].text.replace('\n', '')
                }
                results.append(result)

        #先保存一份没有做检索的模型输出结果result
        # 将result保存下来
        if not os.path.exists(dpo_data_dir):
            os.makedirs(dpo_data_dir)
        multi_output_data_path = os.path.join(dpo_data_dir, "dpo_"+sub_data_name+"_wo_retrieval.json")    
        with open(multi_output_data_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    if(args.max_num > -1):
        results = results[:args.max_num]    
    # 对推理结果进行检索测试，获得检索得分，形成一个字典列表，为[{"qid":"----", "instruction":"---","gt_output":"---", "pred_output":["---", "---", "---"...], "retriever_score":["--","--","--"...]， "sub_dataset":"--"}]

    # 针对每一个改写后的句子进行检索测试,获得4组得分,然后把4组得分合为一体,记为一个二维数组 [[...],[...],[...],[...]]
    # 执行检索

    ir_corpus,ir_relevant_docs,id_corpus_emb_dict,reward_model,bm25,tf_idf = initialize_retriever(
                model_path = retriever_model_path, 
                retriever_type=args.retriever_type,
                output_dir=os.path.join(retrieval_data_dir, sub_data_name+"_ins", "fuzzy"),
                phase="train",
            )
    # 保存一个query docs pairs list，其各项内容为{qid: {query: ---, related_docs: [---, ---, ---...]}}
    query_docs_pairs_list = []
    for res in results:
        
        qid = res["qid"]
        instruction = res["instruction"]
        gt_output = res["gt_output"]
        related_docs_id = ir_relevant_docs[int(qid)]
        related_docs = []
        for doc_id in related_docs_id:
            related_docs.append(ir_corpus[doc_id])
        query_docs_pairs_list.append({"qid":qid, "query": instruction, "related_docs":related_docs, "gt_output": gt_output})
        

    grpo_data_path = os.path.join(dpo_data_dir, "dpo_"+sub_data_name+"_for_grpo.json")    
    with open(grpo_data_path, 'w', encoding='utf-8') as f:
        json.dump(query_docs_pairs_list, f, ensure_ascii=False, indent=4)

    query_score_list = []
    for i in range(len(results)):
        query_score_list.append([])
    for i in range(args.n):
        query_list,qid_list = [],[]
        for res in results:
            query_list.append(res["pred_output"][i])
            qid_list.append(res["qid"])
        query_emb_dir = os.path.join(retrieval_data_dir, sub_data_name+"_ins", "fuzzy",args.retriever_type,"predict_"+str(i))
        if not os.path.exists(query_emb_dir):
            os.makedirs(query_emb_dir)

        ir_evaluator=APIEvaluator(query_list,qid_list,ir_corpus,ir_relevant_docs,id_corpus_emb_dict,reward_model,bm25,tf_idf,args.retriever_type,query_emb_dir,write_csv=True,phase='train')
        score_list = ir_evaluator()
        for s, score in enumerate(score_list):    
            query_score_list[s].append(score["ndcg"]/2)
    
    final_results = []
    for r in range(len(results)):
        results[r]["retriever_score"] = query_score_list[r]
        # 根据检索分数的最大最小值,构建dpo
        max_value = max(query_score_list[r])
        max_index = query_score_list[r].index(max_value)

        min_value = min(query_score_list[r])
        min_index = query_score_list[r].index(min_value)
        #真的存在高低分的时候
        if(max_value > min_value):
            results[r]["chosen"] = results[r]["pred_output"][max_index]
            results[r]["rejected"] = results[r]["pred_output"][min_index]
            final_results.append(results[r])
    # 将result保存下来
    if not os.path.exists(dpo_data_dir):
        os.makedirs(dpo_data_dir)
    dpo_data_path = os.path.join(dpo_data_dir, "dpo_"+sub_data_name+"_retrieval_by_"+args.retriever_type+".json")    
    with open(dpo_data_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("dpo_"+sub_data_name+"_retrieval_by_"+args.retriever_type + "当前数据集的数据量为：")
    print(len(final_results))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run LLM with specified parameters")
    parser.add_argument("--sub_data_name",default='G3',type=str)
    parser.add_argument("--retriever_type",default='bm25',type=str)
    parser.add_argument("--gpu_id",default='4',type=str)
    parser.add_argument("--max_num",default=-1,type=int)
    # bridge model推理参数
    parser.add_argument('--bridge_model_dir', type=str, default=None, help='Path of bridge model')
    parser.add_argument('--bridge_model_path', type=str, default=None, help='Path of bridge model')
    parser.add_argument('--sft_data_dir', type=str, default='{project_root}/data/sft_bridge_model_train_test200', help='SFT数据集位置')

    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of parallel tensor slices')
    parser.add_argument('--temperature', type=float, default=1, help='Sampling temperature')
    parser.add_argument('--n', type=int, default=4, help='Number of output sequences to generate')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate')
    parser.add_argument('--template',type=str,default='llama3')
    parser.add_argument('--gpu',type=float,default=0.4)
    # 检索参数
    parser.add_argument("--data_dir", default="{project_root}/data/retrieval/G1_cat/hybrid", type=str,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--retriever_model_path", default="{project_root}/models/ToolBench_IR_bert_based_uncased", type=str)
    parser.add_argument("--max_length", default=256, type=int,)
    parser.add_argument("--phase", default="test", type=str,)
    parser.add_argument("--retrieval_data_dir",default='{project_root}/data/retrieval-test200',type=str, help="用于检索的数据位置，其中可获得待查集")
    parser.add_argument("--dpo_data_dir",default="{project_root}/data/dpo_bridge_model_train_test200",type=str)
    args = parser.parse_args()
    
    
    main(args)
    # 批量
    # retriever_types = ["ToolRetriever"]
    # # retriever_types = ["bm25", "TF-IDF","ToolRetriever"]
    # # retriever_types = ["ada"]
    # for r_type in retriever_types:
    #     args.retriever_type = r_type
    #     main(args)
