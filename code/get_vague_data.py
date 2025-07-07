import argparse
from src.llm import ChatGpt,fuzzy_template
import json
from tqdm import tqdm
import os
import string
def create_fuzzy_data(args):
    model =  ChatGpt(args.model_name, args.api_key, args.api_url)
    
    exist_data = []
    # for phase in ['train', 'test']:
    for phase in ['test']:
        # fuzzy_txt = os.path.join(args.data_dir,'fuzzy_' + args.model_name, f'{phase}.query.txt')
        fuzzy_txt = os.path.join(args.data_dir,'fuzzy', f'{phase}.query.txt')
        os.makedirs(os.path.dirname(fuzzy_txt), exist_ok=True)
        
        hybrid_txt = os.path.join(args.data_dir,'hybrid',f'{phase}.query.txt')
        hybrid_json = os.path.join(args.data_dir,'hybrid',f'{phase}.json')
        with open(hybrid_txt,'r') as f:
            hybrid_queries = f.readlines()
            query_qid_dict = {}
            for d in hybrid_queries:
                id,query = int(d.split('\t')[0].strip()), d.split('\t')[1].strip().translate(str.maketrans('', '', string.punctuation))
                query_qid_dict[query] = id

        with open(hybrid_json,'r') as f:
            hybrid_data = json.load(f)
        if os.path.exists(fuzzy_txt): 
            with open(fuzzy_txt, 'r') as f:
                orig_data = f.readlines()
                for d in orig_data:
                    id = int(d.split('\t')[0].strip())
                    exist_data.append(id)
        else:
            print(f"Warning: {fuzzy_txt} does not exist. Creating a new file.")
            with open(fuzzy_txt, 'w') as f:
                pass

        for i in tqdm(range(0,len(hybrid_data),args.batch_size)):
            batch_data = hybrid_data[i:i+args.batch_size]
            batch_prompt = []
            batch_id = []
            id_query_dict = {} 
            for d in batch_data:
                # d['query'] = d['query'].strip().strip('"').replace('""', '"')

                if isinstance(d['query'],list):
                    d['query'] = d['query'][0]
                if query_qid_dict[d['query'].strip().translate(str.maketrans('', '', string.punctuation))] in exist_data:
                    continue
                else:
                    id = query_qid_dict[d['query'].strip().translate(str.maketrans('', '', string.punctuation))]
                    exist_data.append(id)
                    
                    tool_list_str = ''
                    for api in d['api_list']:
                        api_identity = [api['tool_name'], api['api_name']]
                        if api_identity in d['relevant APIs']:
                            tool_list_str += f"[tool_name:{api['tool_name']},api_name:{api['api_name']}],"
                    
                    tool_list_str=tool_list_str[:-1]
                    batch_prompt.append(fuzzy_template.format(instruction=d['query'], apis=tool_list_str))
                    batch_id.append(query_qid_dict[d['query'].strip().translate(str.maketrans('', '', string.punctuation))])
            
            model.fuzzy_generate(batch_prompt,batch_id,id_query_dict)
            with open(fuzzy_txt,'a') as f:
                for qid,fuzzy_query in id_query_dict.items():
                    # 偶现输出换行情况,排除
                    fuzzy_query = fuzzy_query.replace("\n"," ")
                    f.write(f"{int(qid)}\t{fuzzy_query}\n")

        # 将生成结果按id数字大小进行排序，再重新写入
        with open(fuzzy_txt, 'r') as f:
            fuzzy_data = f.readlines()
        fuzzy_data_list = []
        for d in fuzzy_data:
            id, query = int(d.split('\t')[0].strip()), d.split('\t')[1].strip()
            fuzzy_data_list.append({"id":id,"query":query})
        sorted_fuzzy_data_list = sorted(fuzzy_data_list, key=lambda x:x["id"])
        with open(fuzzy_txt,'w') as f:
            for item in sorted_fuzzy_data_list:
                id_, query_ = item["id"], item["query"] 
                f.write(f"{id_}\t{query_}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--data_name', type=str, default='G1_cat', help='input file path')
   
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--data_dir', type=str, default='{project_root}/data/retrieval-test200/G3_ins', help='')
    parser.add_argument('--model_name', type=str, default='', help='model name')
    parser.add_argument('--api_key', type=str, default= '' , help='api key')
    parser.add_argument('--api_url', type=str, default="", help='api url')

    args = parser.parse_args()
    # args.data_dir = os.path.join("{project_root}/data/retrieval-test200/", args.data_name)
    create_fuzzy_data(args)
    print("将 hybrid 文件夹下的其余文件复制到 fuzzy 文件夹下")
    # os.system(f"cp -n {os.path.join(args.data_dir,'hybrid')}/* {os.path.join(args.data_dir,'fuzzy_' + args.model_name)}/")
    os.system(f"cp -rn {os.path.join(args.data_dir,'hybrid')}/* {os.path.join(args.data_dir,'fuzzy')}/")