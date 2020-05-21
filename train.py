#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def train():
    """
    训练步骤：
        1.有标记好的xlsx、csv或其他格式的数据，参考data/sample_files/手工标记好的示例地址.xlsx
        2.使用other文件夹下的preprocessing.py文件生成数据集
        3.在train/helper.py中找到data_dir属性，将其修改为数据集的目录
        4.在train/helper.py中找到output_dir属性,修改该值，否则在此次训练开始后，会自动删除之前训练好的模型
        5.（可选）在train/helper.py中修改有关训练的超参数属性，例如：batch_size,learning_rate等等
    :return:
    """
    # 确保已经在dataset中产生了trian、test、dev文件
    import os
    from train.helper import get_args_parser
    from train.bert_lstm_ner import train

    args = get_args_parser()
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    train(args=args)


if __name__ == '__main__':
    train()