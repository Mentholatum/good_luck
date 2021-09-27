import argparse
import os
import util
import torch



def arg_parse():
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--state_dict', default='checkpoint/resnet18-5c106cde.pth', type=str)
    parser.add_argument('--outpath', default='output/', type=str)
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--checkpoint', default='checkpoint',type = str)
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--cudaID', default='0', type=str, help='gpu device id')

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cudaID
    seed = args.seed
    util.fix_all_seed(seed)