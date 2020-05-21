# -*- coding: utf-8 -*-
# @Author  : woleto
# @Time    : 2020/3/24 16:26


from predict import *

# 预测整个文件
# address_file_path = "data/test/柳州待匹配.xlsx"
# predict_result_path = "data/test/柳州待匹配_result.csv"
# predict_file(address_file_path, predict_result_path)

# 预测一条地址
address_str = "北京市海淀区四道口路大钟寺中坤广场西侧停车场对面北下关中昊全民健身中心"
atom_list, token_list, predict_sequence = predict_offline(address_str)
result = participles_sequence(atom_list, predict_sequence[0])
# print(atom_list, token_list, predict_sequence)
print('输入地址：' + address_str)
print('地址分词结果：' + result)
