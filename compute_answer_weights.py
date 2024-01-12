# Project:
#   VQA
# Description:
#   Script to compute answer weights 
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import torch 
import pickle
import misc.io as io
from misc import dirs
from collections import Counter

torch.manual_seed(1234) # use same seed for reproducibility

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():
    # read config file读取配置文件
    config = io.read_config(args.path_config)

    path_output = jp(config['path_qa'], 'answer_weights')
    dirs.create_folder(path_output)
    path_output_file = jp(path_output, 'w.pt')

    path_input = jp(config['path_qa'], 'processed')
    path_input_file_train = jp(path_input, 'trainset.pickle')
    path_input_file_val = jp(path_input, 'valset.pickle')
    path_input_file_test = jp(path_input, 'testset.pickle')



    # read train set
    with open(path_input_file_train, 'rb') as f1, open(path_input_file_val, 'rb') as f2,open(path_input_file_test, 'rb') as f3:
        data_train = pickle.load(f1)
        data_val = pickle.load(f2)
        data_test = pickle.load(f3)
    data = data_train+data_val+data_test

    # group all answers
    if config['dataset'] == 'gqa':
        answers = [e['ma_index'] for e in data] + [e['sa_index'] for e in data]
    else:
        answers = [e['answer_index'] for e in data]
    countings = Counter(answers).most_common()
    countings_dict = {e[0]:e[1] for e in countings}
    weights = torch.zeros(len(countings_dict))
    for i in range(weights.shape[0]):
        weights[i] = countings_dict[i]

    # normalize weights as suggested in https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514
    weights = 1 - weights/weights.sum()

    # save weights to target file
    torch.save(weights, path_output_file)

if __name__ == '__main__':
    main()