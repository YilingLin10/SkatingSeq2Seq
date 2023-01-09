import time
from argparse import ArgumentParser, Namespace
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from data.dataset import IceSkatingDataset
from model.seq2seq import Transformer
from tqdm import trange, tqdm
from typing import Dict
import json
import os
import csv

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="loop", help="old, loop, flip, all_jum"
    )
    parser.add_argument(
        "--model_path", type=str, help="path to saved model checkpoints"
    )
    args = parser.parse_args()
    return args

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    same_seed(0)
    PAD_IDX = 4
    ######################################################################
    #                            LOAD MODEL                              #
    ######################################################################
    EMB_SIZE = 34
    NHEAD = 2
    FFN_HID_DIM = 128
    BATCH_SIZE = 64
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    SRC_VOCAB_SIZE = 5
    TGT_VOCAB_SIZE = 5
    transformer = Transformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                    EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                    FFN_HID_DIM)

    model = transformer.to(DEVICE)
    ckpt_path = args.model_path + "save/transformer_bin_class.pth"
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    ######################################################################
    #                            LOAD DATA                              #
    ######################################################################
    test_file = "/home/lin10/projects/SkatingJumpClassifier/data/{}/alphapose/test.pkl".format(args.dataset)
    tag2idx_file = "/home/lin10/projects/SkatingJumpClassifier/data/tag2idx_seq2seq.json"
    test_dataset = IceSkatingDataset(pkl_file=test_file, 
                                    tag_mapping_file=tag2idx_file, 
                                    subtract_feature=False)
    test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    ######################################################################
    #                            INFERENCE                              #
    ######################################################################
    # function to generate output sequence using greedy algorithm
    def greedy_decode(model, src, start_symbol):
        src = src.to(DEVICE)
        seq_len = src.size(0)
        memory = model.encode(src)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(seq_len):
            memory = memory.to(DEVICE)
            tgt_mask = (model.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        return ys

    # actual function to translate input sentence into target language
    def translate(model: torch.nn.Module, src):
        model.eval()
        tgt_tokens = greedy_decode(
            model, src, start_symbol=0).flatten()
        return tgt_tokens[1:]

    for video_name, src, tgt in test_dataloader:
        print(video_name)
        print(tgt.transpose(0, 1))
        answer = translate(model, src)
        print(answer)
        break

if __name__ == "__main__":
    args = parse_args()
    main(args)