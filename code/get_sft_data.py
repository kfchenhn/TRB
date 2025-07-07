import os
import json
import copy

# 从hybird和fuzzy两种数据中获得成对数据，用作instruction 和 out        
instruction_template = "Your task is to expand the user's query into a more complete and professional sentence structure. You should only restate or expand on the user's query without providing answers, actions, or attempting to resolve the query. Your response should **strictly** be a rephrased version of the user’s input and should **not** include any additional information, actions, or solutions.\nExamples:\nExample 1:\nOrig query: I'm organizing a company gaming tournament and need player stats for top players in popular games. Can you also recommend some good gaming gear for the event?\nDetailed query: I'm organizing a gaming tournament for my company's employees. Could you provide the statistics and ratings of highly skilled players in popular games like Dota 2 and World of Tanks? Also, recommend some gaming peripherals and accessories for the event.\nExample 2:\nOrig query: I want to make a special playlist for my family. Can you suggest some hit songs from different music styles? Also, give me more info about the playlist I'm putting together. Finally, can you get me the link to a specific track I want to add?\nDetailed query: I want to surprise my family with a personalized playlist. Can you recommend some popular tracks from different genres? Additionally, provide me with the detailed information of a playlist that I want to create. Also, fetch the track URL of a specific song that I want to include in the playlist.\nNow, please expand the user's query:\nOrig query: {query} \nDetailed query: \n"

# instruction来自fuzzy
# output来自hybird
data_dir = "{project_root}/data/retrieval-test200"
sft_dir = os.path.join(os.path.dirname(data_dir), "sft_bridge_model_train_test200")

if not os.path.exists(sft_dir):
    os.makedirs(sft_dir)

all_sft_dataset = {}
for dataset_name in os.listdir(data_dir):
    sft_dataset = []
    sub_data_dir = os.path.join(data_dir, dataset_name)
    fuzzy_data_path = os.path.join(sub_data_dir, "fuzzy", "train.query.txt")
    hybird_data_path = os.path.join(sub_data_dir, "hybrid", "train.query.txt")
    # 构建sft数据集
    fuzzy_query_dict = {}
    hybird_query_dict = {}
    with open(fuzzy_data_path,'r') as f:
        fuzzy_queries = f.readlines()
        for d in fuzzy_queries:
            id,query = d.split('\t')[0].strip(), d.split('\t')[1].strip()
            fuzzy_query_dict[id] = query

    with open(hybird_data_path,'r') as f:
        hybird_queries = f.readlines()
        for d in hybird_queries:
            id,query = d.split('\t')[0].strip(), d.split('\t')[1].strip()
            hybird_query_dict[id] = query

    for id, fuzzy_query in fuzzy_query_dict.items():
        hybird_query = hybird_query_dict[id]

        item = {
            "instruction": instruction_template.format(query=fuzzy_query),
            "output": hybird_query,
            "id": id,
            "sub_dataset": dataset_name

        }
        sft_dataset.append(item)
    
    sft_path = os.path.join(sft_dir, dataset_name+"_trainset.json")
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=4)

    all_sft_dataset[dataset_name] = sft_dataset

# 将6个子集合并成三个
sub_dataset_name = ["G1_ins","G1_cat","G1_tool","G2_ins","G2_cat","G3_ins"]
merge_dataset_name = ["G1","G2"]
for merge_name in merge_dataset_name:

    if merge_name == "G1":
        ins_datalist = all_sft_dataset[merge_name+"_ins"]
        cat_datalist = all_sft_dataset[merge_name+"_cat"]
        tool_datalist = all_sft_dataset[merge_name+"_tool"]
        # 根据各自的"output"确认是否可以合并
        merge_datalist = []
        ins_outputs = [data["output"] for data in ins_datalist]
        cat_outputs = [data["output"] for data in cat_datalist]
        tool_outputs = [data["output"] for data in tool_datalist]
        for i_, output in enumerate(ins_outputs):
            if output in cat_outputs and output in tool_outputs:
                merge_datalist.append(ins_datalist[i_])
        
        merge_sft_path = os.path.join(sft_dir, merge_name+"_trainset.json")
        with open(merge_sft_path, "w", encoding="utf-8") as f:
            json.dump(merge_datalist, f, ensure_ascii=False, indent=4)
        print("G1_ins 训练集数量: ", len(ins_datalist))
        print("G1_cat 训练集数量: ", len(cat_datalist))
        print("G1_tool 训练集数量: ", len(tool_datalist))
        print("G1_merge 训练集数量: ", len(merge_datalist))

    if merge_name == "G2":
        ins_datalist = all_sft_dataset[merge_name+"_ins"]
        cat_datalist = all_sft_dataset[merge_name+"_cat"]
        # 根据各自的"output"确认是否可以合并
        merge_datalist = []
        ins_outputs = [data["output"] for data in ins_datalist]
        cat_outputs = [data["output"] for data in cat_datalist]
        for i_, output in enumerate(ins_outputs):
            if output in cat_outputs:
                merge_datalist.append(ins_datalist[i_])
        
        merge_sft_path = os.path.join(sft_dir, merge_name+"_trainset.json")
        with open(merge_sft_path, "w", encoding="utf-8") as f:
            json.dump(merge_datalist, f, ensure_ascii=False, indent=4)
        print("G2_ins 训练集数量: ", len(ins_datalist))
        print("G2_cat 训练集数量: ", len(cat_datalist))
        print("G2_merge 训练集数量: ", len(merge_datalist))

    if merge_name == "G3":
        ins_datalist = all_sft_dataset[merge_name+"_ins"]
        print("G3_ins 训练集数量: ", len(ins_datalist))
