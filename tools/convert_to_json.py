'''
convert.py is a script that reads the content of the source and target files in the subfolders of the folder_path
and combines them into a dataset. The source file is the input field, and the target file is the output field.

Llama3.1 8B 使用《史记》七十列传文本数据微调训练，实现现代文翻译至古文

Please refer to the youtube video for more details: https://youtu.be/Tq6qPw8EUVg
'''

# import os
import pathlib
import json
import pandas as pd
import logging

root = pathlib.Path(__file__).parent

source_root = root.joinpath('..').resolve()
save_root = root.joinpath("dataset").resolve()

# get all subfolders in the folder, then for each subfolder, get source file and target file,
# then read the content of the files then combine them into a dateset, the source file is the input field,
# the target file is the output field, and the target file is the output field

logLevel = logging.DEBUG
logFile = pathlib.Path(__file__).with_suffix('.log')
# Basic logging configuration
# fmt:off
logging.basicConfig(
    level=logLevel,
    format=('%(message)s' if logLevel == logging.INFO else '%(asctime)s %(filename)s(%(lineno)04d) [%(levelname)-8s]: %(message)s' ),
    handlers=[logging.FileHandler(logFile, mode='w', encoding='utf-8'), logging.StreamHandler()],
    datefmt='%Y-%m-%d %H:%M:%S',
)
# fmt:on


def get_files(folder_path):
    subfolders = [d for d in folder_path.iterdir() if d.is_dir()]
    # print(subfolders)
    dataset = []

    source_file = "source.txt"
    target_file = "target.txt"
    for folder in subfolders:
        source_content = folder.joinpath(source_file).open("r", encoding="utf-8").read().splitlines()
        target_content = folder.joinpath(target_file).open("r", encoding="utf-8").read().splitlines()

        # # source and target needs to be split by "\n"
        # source_content = source_content.splitlines()
        # target_content = target_content.splitlines()

        # source and target should be saved into dateset line by line
        for src, target in zip(source_content, target_content):
            dataset.append([src, target])

    return dataset


def dump_data_file(folder_path):
    dump_path = save_root.joinpath(folder_path.parent.relative_to(source_root))
    dump_path.mkdir(parents=True, exist_ok=True)

    dump_file_name = dump_path.joinpath(f"dataset_{folder_path.name}.jsonl")
    dataset = get_files(folder_path)
    # add one column "instruction" with the content "请把古文翻译成现代汉语" to the dataset
    df = pd.DataFrame(dataset, columns=["source", "target"])
    df["instruction"] = "请把现代汉语翻译成古文"

    # rename the columns: source -> output, target -> input
    df.rename(columns={"source": "output", "target": "input"}, inplace=True)

    # print length of the dataset
    # logging.debug(len(df))

    # save the dataset into a jsonl file
    df.to_json(dump_file_name, orient="records", lines=True, force_ascii=False)


def merge_dataset(data_folder):
    datafiles = sorted(data_folder.rglob('dataset_*.jsonl'))
    books = list(set([p.parent for p in datafiles if '双语' not in p.name]))

    for book in books:
        logging.info(book.name)
        d = []
        for sub_book in book.glob("*.jsonl"):
            df = pd.read_json(sub_book, orient="records", lines=True)
            df['book'] = f"{book.name}/{sub_book.name.replace('dataset_','')}"
            d.append(df)
        pd.concat(d).to_json(book.joinpath(f"full_{book.name}.jsonl"), orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    # folder_path = source_root.joinpath(r"双语数据/史记/七十列传").resolve()
    folders = list(set([p.parent.parent for p in source_root.rglob('bitext.txt')]))

    for folder in folders:
        logging.info(f'{folder.relative_to(source_root)}')
        dump_data_file(folder)

    merge_dataset(save_root)
