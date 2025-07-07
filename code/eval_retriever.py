import argparse
import os 
from src.utils import initialize_retriever,read_query
from src.api_evaluator import APIEvaluator
import json
import pandas as pd
# os.environ["CUDA_HOME"] = "/usr/local/cuda-11.1"
# os.environ["CUDA_PATH"] = "/usr/local/cuda-11.1"
project_root="******************"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", default="{project_root}/data/retrieval-test200/G3_ins/hybrid", type=str,help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--model_path", default="{project_root}/models/ToolBench_IR_bert_based_uncased", type=str)
    parser.add_argument("--max_length", default=256, type=int,)
    parser.add_argument("--phase", default="test", type=str,)
    parser.add_argument("--retriever_type",default='ToolRetriever',type=str) # bm25, TF-IDF, ada, ToolRetriever
    parser.add_argument("--train_type",default='dpo',type=str)
    parser.add_argument('--gpu_id',type=str, default="4")
    # 批量测试参数
    parser.add_argument("--data_root", default="{project_root}/data/retrieval-test200-grpo_0629", type=str, help="The input data root")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    # retriever_types = ["bm25"]
    retriever_types = ["bm25", "TF-IDF", "ada", "ToolRetriever"]
    # 遍历所有的retriever type
    # 遍历所有的数据目录（toolbench, VG, TRB）
    ####批量#####
    # for r_type in retriever_types:
    # args.retriever_type = r_type
    r_type = args.retriever_type
    for sub_data_name in os.listdir(args.data_root):
    # for sub_data_name in ["G3_ins"]:
        for sub2_data_name in os.listdir(os.path.join(args.data_root, sub_data_name)):
        # for sub2_data_name in ["hybrid"]:
            args.data_dir = os.path.join(args.data_root, sub_data_name, sub2_data_name)
            ####批量#####
            if  args.train_type == "dpo" and r_type not in sub2_data_name:
                continue 
            output_dir = os.path.join(args.data_dir, args.retriever_type)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # args.data_dir = output_dir
            ir_corpus,ir_relevant_docs,id_corpus_emb_dict,reward_model,bm25,tf_idf = initialize_retriever(
                        args.model_path, 
                        retriever_type=args.retriever_type,
                        output_dir=args.data_dir,
                        phase=args.phase,
                    )
            data_path = os.path.join(args.data_dir, f"{args.phase}.query.txt")
            qid_list,query_list = read_query(data_path)
            ir_evaluator=APIEvaluator(query_list,qid_list,ir_corpus,ir_relevant_docs,id_corpus_emb_dict,reward_model,bm25,tf_idf,args.retriever_type,output_dir,write_csv=True,phase='test')
            score_list = ir_evaluator()
            with open(os.path.join(output_dir, f'{args.phase}_score.json'), 'w') as f:
                json.dump(score_list, f, indent=4)
                print(os.path.join(output_dir, f'{args.phase}_score.json'))


    # summary result
    result = []
    summary_result = os.path.join(args.data_root, "summary_test_result.json")
    for data_name in os.listdir(args.data_root):
        
        data_result = []
        data_dir = os.path.join(args.data_root, data_name)
        for sub_data_name in os.listdir(data_dir):
            
            sub_data_result = []
            sub_data_dir = os.path.join(data_dir, sub_data_name)
            # if sub_data_name.split("_")[0] == "fuzzy":
            for retriever_type in retriever_types:
                if args.train_type=="dpo" and retriever_type not in sub_data_name:
                    continue
                test_result_path = os.path.join(sub_data_dir, retriever_type, "test_Retrieval_evaluation_results.csv")

                df =  pd.read_csv(test_result_path)
                keys = df.columns.tolist()[2:]
                values = df.iloc[-1].tolist()[2:]
                ndcg_result = {}
                for i in range(len(keys)):
                    key = keys[i]
                    value = values[i].replace("[", "").replace("np.float64(", "").replace(")","").replace("]","").strip()
                    value = round(float(value)*100,2)
                    ndcg_result[key] = value
                result_path = os.path.join(sub_data_dir, retriever_type)

                path_info = result_path.split("/")

                d_name, rewrite_type, r_name = path_info[-3], path_info[-2], path_info[-1]
                
                # 如果是多个checkpoint测试，然后选最好的
                checkpoint_idx="single"
                if "checkpoint" in sub_data_name:
                    checkpoint_idx = sub_data_name.split("_")[-1]
                    # d_name=d_name[:-(len(checkpoint_idx)+1)]
            
                sub_data_result.append({
                    "checkpoint_idx": checkpoint_idx,
                    "dataset_name": d_name,
                    "retriever": r_name,        
                    "ndcg": ndcg_result
                })

            #         {
            #             'fuzzy_deepseek-chat_rewrite_by_deepseek':['bm25':['G1_ins':{"ndcg": {"Average NDCG@5": "0.6407452889978562","Average NDCG@10": " 0.6927852397938669"
            # }}]],

            #         }
            
            result_item = {
                sub_data_name: sub_data_result
            }
            data_result.append(result_item)
        
        
        # 如果是多个checkpoint测试，然后选最好的
        if args.train_type=="dpo""checkpoint" in os.listdir(data_dir)[0]:
            new_key_max_score_dict = {}
            for result_item in data_result:
                item_key = result_item.keys()[0]
                if "checkpoint" in item_key:
                    checkpoint_idx = item_key.split("_")[-1]
                    new_item_key=item_key[:-(len(checkpoint_idx)+1)]
                    if not new_item_key in new_key_max_score_dict:
                        new_key_max_score_dict[new_item_key] = list(result_item.values())[0]
                    else:
                        if list(result_item.values())[0][0]["ndcg"]["Average NDCG@5"] > new_key_max_score_dict[new_item_key][0]["ndcg"]["Average NDCG@5"]:
                            new_key_max_score_dict[new_item_key] = list(result_item.values())[0]
            result.append({
                data_name: new_key_max_score_dict
            })        
        else:

            result.append({
                data_name: data_result
            })

    with open(summary_result,"w",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False, indent=4)


