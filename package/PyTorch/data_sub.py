import pandas as pd
import os
import shutil


def choice_sample(sample, sorce_dir, to_dir):
    label_list = pd.read_csv(os.path.join(sorce_dir, "labels.csv"))
    label_list = label_list.sample(n=sample)

    new_data_path = []
    new_label = []
    for sub_path, sub_label in zip(label_list["data_path"], label_list["label"]):
        source_file = os.path.join(sorce_dir, sub_path)
        target_file = os.path.join(to_dir, sub_path)

        abs_file_path = target_file.replace(os.path.basename(target_file), "")

        if not os.path.exists(abs_file_path):
            os.makedirs(abs_file_path)
        shutil.copy(source_file, target_file)
        new_data_path.append(sub_path)
        new_label.append(sub_label)

    pd.DataFrame({"data_path": new_data_path, "label": new_label}).to_csv(os.path.join(to_dir, "labels.csv"),
                                                                          index=False)


if __name__ == '__main__':
    sample = 1000
    sorce_dir = "/Users/tianjian/Projects/datasets/imgs/"
    to_dir = "/Users/tianjian/Projects/datasets/imgs/sub_data/"
    choice_sample(sample, sorce_dir, to_dir)
