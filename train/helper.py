# -*- coding: utf-8 -*-
import argparse
import logging
import os

__all__ = ['get_args_parser', 'set_logger']


def get_args_parser():
    from .bert_lstm_ner import __version__
    parser = argparse.ArgumentParser()
    if os.name == 'nt':
        # chinese_L-12_H-768_A-12为google发布的预训练模型
        bert_path = 'bert_base/chinese_L-12_H-768_A-12'
        root_path = r''
    else:
        bert_path = '/home/chinese_L-12_H-768_A-12/'
        root_path = '/home/BERT-BiLSTM-CRF-NER'

    group1 = parser.add_argument_group('File Paths', 'config the path, checkpoint and filename of a pretrained/fine-tuned BERT model')

    # 1.data_dir是存放最终数据集的地方，使用自己数据训练一定记得确认数据集是否在如下目录（当前是：data/dataset）
    group1.add_argument('-data_dir', type=str, default=os.path.join(root_path, 'data/dataset'),
                        help='train, dev and test data dir')

    # 2.output_dir是放模型以及中间数据存放的目录，如果要生成新模型，一定要确认，否则会抹掉之前训练好的模型
    group1.add_argument('-output_dir', type=str, default=os.path.join(root_path, 'output'),
                        help='directory of a pretrained BERT model')

    group1.add_argument('-bert_path', type=str, default=bert_path)
    group1.add_argument('-bert_config_file', type=str, default=os.path.join(bert_path, 'bert_config.json'))
    group1.add_argument('-init_checkpoint', type=str, default=os.path.join(bert_path, 'bert_model.ckpt'),
                        help='Initial checkpoint (usually from a pre-trained BERT model).')
    group1.add_argument('-vocab_file', type=str, default=os.path.join(bert_path, 'vocab.txt'),
                        help='')

    group2 = parser.add_argument_group('Model Config', 'config the model params')

    # 3.地址的最大长度，当前为100。可以根据需要调整。
    # 例如："北京市朝阳区" 长度为6
    group2.add_argument('-max_seq_length', type=int, default=100,
                        help='The maximum total input sequence length after WordPiece tokenization.')
    group2.add_argument('-do_train', action='store_false', default=True,
                        help='Whether to run training.')
    group2.add_argument('-do_eval', action='store_false', default=True,
                        help='Whether to run eval on the dev set.')
    group2.add_argument('-do_predict', action='store_false', default=True,
                        help='Whether to run the predict in inference mode on the test set.')

    # 4. 一次添加多少条地址到模型进行训练，一般可以选择2的倍数
    group2.add_argument('-batch_size', type=int, default=32,
                        help='Total batch size for training, eval and predict.')
    # 5. 训练学习率
    group2.add_argument('-learning_rate', type=float, default=1e-5,
                        help='The initial learning rate for Adam.')
    # 6.一次训练训练多少代（把所有训练数据从头到尾训练一次是一代）
    group2.add_argument('-num_train_epochs', type=float, default=10,
                        help='Total number of training epochs to perform.')


    group2.add_argument('-dropout_rate', type=float, default=0.5,
                        help='Dropout rate')
    group2.add_argument('-clip', type=float, default=0.5,
                        help='Gradient clip')
    group2.add_argument('-warmup_proportion', type=float, default=0.1,
                        help='Proportion of training to perform linear learning rate warmup for '
                             'E.g., 0.1 = 10% of training.')

    # 7. LSTM网络中一层的神经元数目
    group2.add_argument('-lstm_size', type=int, default=128,
                        help='size of lstm units.')
    # 8. LSTM设置几层（数目较大会造成训练缓慢）
    group2.add_argument('-num_layers', type=int, default=1,
                        help='number of rnn layers, default is 1.')
    group2.add_argument('-cell', type=str, default='lstm',
                        help='which rnn cell used.')

    # 9. 训练多少步，保存一次模型
    group2.add_argument('-save_checkpoints_steps', type=int, default=500,
                        help='save_checkpoints_steps')
    # 10.训练多少步，保存一次当前状态
    group2.add_argument('-save_summary_steps', type=int, default=500,
                        help='save_summary_steps.')
    group2.add_argument('-filter_adam_var', type=bool, default=False,
                        help='after training do filter Adam params from model and save no Adam params model in file.')
    group2.add_argument('-do_lower_case', type=bool, default=True,
                        help='Whether to lower case the input text.')
    group2.add_argument('-clean', type=bool, default=True)
    group2.add_argument('-device_map', type=str, default='0',
                        help='witch device using to train')

    # add labels
    group2.add_argument('-label_list', type=str, default=None,
                        help='User define labels， can be a file with one label one line or a string using \',\' split')

    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    parser.add_argument('-ner', type=str, default='ner', help='which modle to train')
    parser.add_argument('-version', action='version', version='%(prog)s ' + __version__)
    return parser.parse_args()


def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger