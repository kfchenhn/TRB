import json
import argparse
import pickle
import pandas as pd
from vllm import LLM, SamplingParams
import os

rewrite_instruction_template = "Your task is to expand the user's query into a more complete and professional sentence structure. You should only restate or expand on the user's query without providing answers, actions, or attempting to resolve the query. Your response should **strictly** be a rephrased version of the user’s input and should **not** include any additional information, actions, or solutions.\nExamples:\nExample 1:\nOrig query: I'm organizing a company gaming tournament and need player stats for top players in popular games. Can you also recommend some good gaming gear for the event?\nDetailed query: I'm organizing a gaming tournament for my company's employees. Could you provide the statistics and ratings of highly skilled players in popular games like Dota 2 and World of Tanks? Also, recommend some gaming peripherals and accessories for the event.\nExample 2:\nOrig query: I want to make a special playlist for my family. Can you suggest some hit songs from different music styles? Also, give me more info about the playlist I'm putting together. Finally, can you get me the link to a specific track I want to add?\nDetailed query: I want to surprise my family with a personalized playlist. Can you recommend some popular tracks from different genres? Additionally, provide me with the detailed information of a playlist that I want to create. Also, fetch the track URL of a specific song that I want to include in the playlist.\nNow, please expand the user's query:\nOrig query: {orig_query} \nDetailed query: \n"



# 读取query.txt，并将其存为一个字典列表
def read_queries(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 按第一个制表符分割
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    qid, query = parts
                    result.append({'qid': qid, 'query': query})
    return result




def main(args):
    # Define sampling parameters from the parsed arguments
    sampling_params = SamplingParams(temperature=args.temperature, n=args.n, max_tokens=args.max_tokens)

    # Initialize LLM with model path and parallelization setting
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tokenizer=args.model_path,
        dtype='bfloat16',
        tokenizer_mode='auto',
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu,
        max_num_seqs=128
    )

    # Read input data from JSON file
    prompts = []
    test_data = read_queries(args.input_file)
    for data in test_data:
        qid = data["qid"]
        query = data["query"]
        text = rewrite_instruction_template.format(orig_query=query)

        if args.template == 'llama2':
                prompts.append(f"<s> [INST] {text} [/INST]")
        elif args.template == 'llama3':
            prompts.append(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        elif args.template == 'alpaca':
            prompts.append(f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{text}\n\n### Response:\n")
    print(len(prompts))
    outputs = llm.generate(prompts, sampling_params)
    print(outputs)
    print(outputs[0])

    results = []
    for i in range(len(test_data)):
        data_item = test_data[i]
        qid = data_item["qid"]
        query = outputs[i].outputs[0].text.replace('\n', '')  
        results.append({"qid": qid, "query": query})

    if not os.path.exists(os.path.dirname(args.output_file)):
        os.mkdir(os.path.dirname(args.output_file))
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(f"{item['qid']}\t{item['query']}\n")

    print("将改写前的文件夹下的其余文件复制到改写后的文件夹下")
    os.system(f"cp -nr {os.path.dirname(args.input_file)}/* {os.path.dirname(args.output_file)}/")

    print(f"Generated outputs saved to {args.output_file}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run LLM with specified parameters")
    parser.add_argument('--input_file', type=str, default='{project_root}/data/retrieval/G1_cat/fuzzy_deepseek-chat/test.query.txt', help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, default='{project_root}/data/retrieval/G1_cat/rewrite_fuzzy_deepseek-chat/test.query.txt', help='Path to save the output file')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of parallel tensor slices')
    parser.add_argument('--temperature', type=float, default=0, help='Sampling temperature')
    parser.add_argument('--n', type=int, default=1, help='Number of output sequences to generate')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate')
    parser.add_argument('--template',type=str, default='llama3')
    parser.add_argument('--gpu',type=float, default=0.8)
    parser.add_argument('--gpu_id',type=str, default="4")

    # #批量测试参数 每个子集训练一个模型，共三个模型
    # parser.add_argument('--model_dir', type=str, default='{project_root}/models/my_bridge_model/sft_0613', help='Path to the multi_model')
    # parser.add_argument('--data_dir', type=str, default='{project_root}/data/retrieval-test200', help='Path to the multi_model')
    
    # args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    # # 批量测试
    # for dataset_name in os.listdir(args.model_dir):
    #     args.model_path = os.path.join(args.model_dir, dataset_name)
    #     for sub_dataset_name in os.listdir(args.data_dir):
    #         if sub_dataset_name.split("_")[0] == dataset_name:
    #             args.input_file = os.path.join(args.data_dir, sub_dataset_name, "fuzzy", "test.query.txt")
    #             args.output_file = os.path.join(args.data_dir, sub_dataset_name, "fuzzy_rewrite_by_"+args.model_dir.split("/")[-1], "test.query.txt")
    #             main(args)


    #批量测试参数，只训练1个sft模型
    parser.add_argument('--model_path', type=str, default='{project_root}/models/my_bridge_model/sft_g123_bridge_model_0625', help='Path to the multi_model')
    parser.add_argument('--data_dir', type=str, default='{project_root}/data/retrieval-test200', help='Path to the multi_model')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    
    # 批量测试
    for sub_dataset_name in os.listdir(args.data_dir):
        args.input_file = os.path.join(args.data_dir, sub_dataset_name, "fuzzy", "test.query.txt")
        args.output_file = os.path.join(args.data_dir, sub_dataset_name, "fuzzy_rewrite_by_"+args.model_path.split("/")[-1], "test.query.txt")
        main(args)
