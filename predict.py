# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/3/4 12:02
from other.predict_base import predict_one, participles_sequence
from other.preprocessing import atomization
import pandas as pd


def predict_address(address):
    predict_sequence = predict_one(address)
    atom_address = atomization(address)
    participles_address = participles_sequence(atom_address, predict_sequence[0])
    return participles_address


def predict_file_address(origin_file_path, result_file):
    """
    Note：
        示例代码，可根据文件本身调节
    :param origin_file_path:
    :param result_file:
    :return:
    """
    address_file_path = origin_file_path
    predict_result_path = result_file
    data = pd.read_csv(address_file_path, sep=',', encoding="gbk", header=None)
    # print(data)
    data["序列标注"] = ""
    data["分词情况"] = ""
    for index, row in data.iterrows():
        if index == 0:
            continue
        try:
            address = row[0]
            atom_address = atomization(address)
            predict_sequence = predict_one(address)
            data.iloc[index, 2] = ','.join(predict_sequence[0])
            # 用序列映射地址，从而进行分词
            participles_address = participles_sequence(atom_address, predict_sequence[0])
            data.iloc[index, 1] = participles_address
        except:
            print(index)
            continue
        if index % 100 == 0:
            data.to_csv(predict_result_path, index=False, header=None)
    data.to_csv(predict_result_path, index=False, header=None)


if __name__ == "__main__":
    # 预测单个地址
    result = predict_address("北京市朝阳区酒仙桥北路甲10号电子城IT产业园107楼6层")
    print(result)

    # 预测整个文件
    # predict_file_address('data/sample_files/展示预测的示例地址文件.csv','data/sample_files/展示预测的示例地址文件结果.csv')

