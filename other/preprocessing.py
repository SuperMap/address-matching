# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/3/16 11:04
import os
import re
import string
import pandas as pd

from bert_base.bert import tokenization
from train.helper import get_args_parser

args = get_args_parser()


def atomization(text):
    atom_list = list()
    my_re = re.compile(r'([a-z0-9]+[-_()]*[a-z0-9]*)', re.I)
    parentheses_re = re.compile(r'[(](.*)[)]', re.S)
    # used '0' replace NaN before, judge it now
    if text == '0':
        return atom_list
    while len(text) > 0:
        if ('\u4e00' <= text[0] <= '\u9fff') or (text[0] in string.punctuation):
            atom_list.append(text[0])
            text = text[1:]
        elif text[0] == '(':
            element = re.search(parentheses_re, text).group(0)
            # atom_list[-1] = atom_list[-1] + element
            text = text[len(element):]
        else:
            try:
                id_tuple = re.search(my_re, text).span()
                start_index = id_tuple[0]
                end_index = id_tuple[1]
                atom_list.append(text[start_index:end_index])
                text = text[end_index:]
            except AttributeError:
                atom_list.append(text[0])
                text = text[1:]
    return atom_list


def tokenized(atomization_list):
    """
    second step, process data that after atomization()
    :param atomization_list:
    :return:
    """
    tokenized_list = []
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    for element in atomization_list:
        if element == '(':
            break
        token_list = tokenizer.tokenize(element)
        tokenized_list.append(token_list[0])
    return tokenized_list


def get_BMSE_by_length(length):
    if length <= 0:
        return list()
    elif length == 1:
        return list(["S"])
    elif length == 2:
        return list(["B", "E"])
    else:
        bmse_list = list(["B"])
        for index in range(length - 2):
            bmse_list.append("M")
        bmse_list.append("E")
        return bmse_list


class AddressDataset(object):
    def __init__(self, file_path, dataset_save_dir):
        self.file_path = file_path
        self.dataset_save_dir = dataset_save_dir
        self.labels = ['province', 'city', 'county', 'town', 'road', 'landmark', 'number', 'poi', 'orient', 'conj',
                       'punc']
        self.dataset_split = [0.6, 0.2, 0.2]

    def generate(self):
        if self.file_path.find('.xl') > 0:
            file_data = pd.read_excel(self.file_path)
        elif self.file_path.endswith('.csv'):
            file_data = pd.read_csv(self.file_path)
        else:
            raise Exception('不支持的文件类型')
        file_data = pd.read_excel(self.file_path)
        model = 'train'
        data_row_sum = len(file_data)
        max_train_index = data_row_sum * self.dataset_split[0]
        max_dev_index = data_row_sum * (self.dataset_split[0] + self.dataset_split[1])

        file_data['BMSE_Label'] = ''
        for index, row in file_data.iterrows():
            # change model
            if max_train_index <= index < max_dev_index:
                model = 'dev'
            if index >= max_dev_index:
                model = 'test'

            split_address_str = str(file_data.iloc[index, 0])
            split_num_label_str = str(file_data.iloc[index, 1])
            split_address = split_address_str.split('|')
            split_num_label = split_num_label_str.split('|')
            # process address have only one segment. it will cause split_address's length is 1.
            if len(split_address) == 0:
                split_address = file_data.iloc[index, 0]
                split_num_label = file_data.iloc[index, 1]
            assert len(split_address) == len(split_num_label), 'the index of' + str(
                index) + ':the length of data not equals label'
            # make label index to label name. for example: 2-->city
            split_label = [int(x) - 1 for x in split_num_label]
            row_data_tokenizer = []
            bmse_label_list = []
            for segment_index in range(len(split_address)):
                address_segment = split_address[segment_index]
                address_segment_label = split_label[segment_index]
                # atomization
                atomization_list = atomization(address_segment)
                # toke
                token_list = tokenized(atomization_list)
                for element in token_list:
                    row_data_tokenizer.append(element)
                print(token_list)
                # get bmse label
                segment_bmse_list = get_BMSE_by_length(len(token_list))
                for segment_bmse in segment_bmse_list:
                    bmse_label_list.append(segment_bmse + '-' + str(self.labels[address_segment_label]))
            # process.write_data_label(row_data_tokenizer, bmse_label_list, model)
            print(row_data_tokenizer, bmse_label_list)
            path = os.path.join(self.dataset_save_dir, model + '.txt')
            with open(path, 'a+', encoding='UTF-8') as w:
                for index in range(len(row_data_tokenizer)):
                    str_line = row_data_tokenizer[index] + ' ' + bmse_label_list[index]
                    w.write(str_line + '\n')
                w.write('\n')


if __name__ == "__main__":
    process = AddressDataset(file_path='data/sample_files/手工标记好的示例地址.xlsx', dataset_save_dir='data')
    process.generate()
