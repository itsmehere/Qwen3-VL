import re

# Define placeholders for dataset paths
RLBENCH_ICL_10_FULL_TRAIN = {
    "annotation_path": "/home/mrao/DarrellGroup/Qwen2.5-VL/data/rlbench_icl_small/anns/train_icl-10_full.json",
    "data_path": "/home/mrao/DarrellGroup/Qwen2.5-VL/data/rlbench_icl_small/train",
}

RLBENCH_ICL_10_FULL_VAL = {
    "annotation_path": "/home/mrao/DarrellGroup/Qwen2.5-VL/data/rlbench_icl_small/anns/val_icl-10_full.json",
    "data_path": "/home/mrao/DarrellGroup/Qwen2.5-VL/data/rlbench_icl_small/train",
}
    
data_dict = {
    "rlbench_icl_10_full_train": RLBENCH_ICL_10_FULL_TRAIN,
    "rlbench_icl_10_full_val": RLBENCH_ICL_10_FULL_VAL,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
