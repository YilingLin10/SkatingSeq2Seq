import time
from argparse import ArgumentParser, Namespace
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_path = f"./experiments/seq2seq/{args.dataset}_{now_time}/"
    save_path = model_path + 'save/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ######################################################################
    #                            LOAD MODEL                              #
    ######################################################################
    NUM_EPOCHS = 15
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

    transformer = transformer.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    def train_epoch(model, optimizer):
        model.train()
        losses = 0
        train_file = "/home/lin10/projects/SkatingJumpClassifier/data/{}/alphapose/train.pkl".format(args.dataset)
        tag2idx_file = "/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json"
        train_dataset = IceSkatingDataset(pkl_file=train_file, 
                                        tag_mapping_file=tag2idx_file, 
                                        subtract_feature=False)
        train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
        
        for _, src, tgt in train_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            # already seq_len first
            tgt_input = tgt[:-1, :]

            tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(tgt)
            tgt_mask, src_padding_mask, tgt_padding_mask = tgt_mask.to(DEVICE), src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)

            logits = model(src, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        return losses / len(train_dataloader)
    
    def evaluate(model):
        model.eval()
        losses = 0
        test_file = "/home/lin10/projects/SkatingJumpClassifier/data/{}/alphapose/test.pkl".format(args.dataset)
        tag2idx_file = "/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json"
        test_dataset = IceSkatingDataset(pkl_file=test_file, 
                                        tag_mapping_file=tag2idx_file, 
                                        subtract_feature=False)
        test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)

        for _, src, tgt in test_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_input = tgt[:-1, :]

            tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(tgt)
            tgt_mask, src_padding_mask, tgt_padding_mask = tgt_mask.to(DEVICE), src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)
            logits = model(src, tgt_input, tgt_mask, src_padding_mask, tgt_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

            ######################################################################
            #                        GENERATE PREDICTION                         #
            ######################################################################
            prob = logits.transpose(0, 1)
            _, batch_preds = torch.max(prob, dim=2)
            labels_expected_padding_mask = (tgt_out==4).transpose(0,1)
            for preds, m, tgt in zip(batch_preds, labels_expected_padding_mask, tgt_out.transpose(0,1)):
                print(tgt[m==False].tolist())
                print(preds[m==False].tolist())
                break
            
        return losses / len(test_dataloader)
    ######################################################################
    #                            TRAIN MODEL                              #
    ######################################################################
    epochs = trange(NUM_EPOCHS, desc="Epoch")
    best_eval_loss = np.inf
    writer = SummaryWriter()
    for epoch in epochs:
        train_loss = train_epoch(transformer, optimizer)
        val_loss = evaluate(transformer)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        writer.add_scalar('TRAIN/LOSS', train_loss, epoch)
        writer.add_scalar('EVAL/Loss', val_loss, epoch)
        if val_loss <= best_eval_loss:
            best_eval_loss = val_loss
            print("SAVING THE BEST MODEL - loss {:.4f}".format(val_loss))
            checkpoint = {
                'epochs': epoch + 1,
                'state_dict': transformer.state_dict(),
            }
            torch.save(checkpoint, save_path + "transformer_bin_class.pth")
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    