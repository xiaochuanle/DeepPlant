# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
import numpy as np
import argparse
import os
import sys
import time
import re

from models_lstm import ModelBiLSTM
from dataloader import Dataset_npy
from dataloader import clear_linecache
from utils.process_utils import display_args
from utils.process_utils import str2bool

from utils.constants_torch import use_cuda


def fix_seeds():
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def train(args):
    total_start = time.time()

    print("[main] train starts..")
    if use_cuda:
        print("GPU is available!")
    else:
        print("GPU is not available!")

    print("reading data..")

    # 获取文件夹下所有的文件和文件夹名
    entries = os.listdir(args.train_file1)
    file_names_1 = [os.path.join(args.train_file1, f) for f in entries if os.path.isfile(os.path.join(args.train_file1, f))]
    file_names_2, file_names_3, file_names_4 = [], [], []
    if args.train_file2 is not None:
        entries = os.listdir(args.train_file2)
        file_names_2 = [os.path.join(args.train_file2, f) for f in entries if os.path.isfile(os.path.join(args.train_file2, f))]
    if args.train_file3 is not None:
        entries = os.listdir(args.train_file3)
        file_names_3 = [os.path.join(args.train_file3, f) for f in entries if os.path.isfile(os.path.join(args.train_file3, f))]
    if args.train_file4 is not None:
        entries = os.listdir(args.train_file4)
        file_names_4 = [os.path.join(args.train_file4, f) for f in entries if os.path.isfile(os.path.join(args.train_file4, f))]

    valid_dataset = Dataset_npy(args.valid_file, kmer=args.seq_len)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4)
    pretrained_model_path = args.pretrained_model
    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*")
            for mfile in os.listdir(model_dir):
                if model_regex.match(mfile):
                    os.remove(model_dir + "/" + mfile)
        model_dir += "/"

    model = ModelBiLSTM(args.seq_len,
                        args.signal_len,
                        args.layernum1,
                        args.layernum2,
                        args.class_num,
                        args.dropout_rate,
                        args.hid_rnn,
                        args.n_vocab,
                        args.n_embed,
                        str2bool(args.is_base),
                        str2bool(args.is_signallen),
                        str2bool(args.is_trace),
                        args.model_type,
                        0)
    # model = torch.compile(model, mode="max-autotune")
    if pretrained_model_path is not None:
        print('Load pretrained model %s' % (pretrained_model_path))
        model.load_state_dict(torch.load(pretrained_model_path))
    if use_cuda:
        model = model.cuda()

    # Loss and optimizer
    weight_rank = torch.from_numpy(np.array([1, args.pos_weight])).float()
    if use_cuda:
        weight_rank = weight_rank.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight_rank)
    if args.optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
    else:
        raise ValueError("optim_type is not right!")
    sys.stdout.flush()
    scheduler = StepLR(optimizer, step_size=2, gamma=0.2)

    # Train the model
    curr_best_f1 = 0
    model.train()
    for epoch in range(args.max_epoch_num):
        all_train_file = file_names_1 + file_names_2 + file_names_3 + file_names_4
        all_train_file = random.sample(all_train_file, len(all_train_file))

        batch_count = 0
        start = time.time()
        curr_best_f1_epoch = 0
        no_best_model = True
        tlosses = []
        for train_file in all_train_file:
            train_dataset = Dataset_npy(train_file, kmer=args.seq_len)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=4)

            for i, sfeatures in enumerate(train_loader):
                kmer, signal, label = sfeatures

                if use_cuda:
                    kmer = kmer.cuda()
                    signal = signal.cuda()
                    label = label.cuda()

                # Forward pass
                outputs, logits = model(kmer, signal)
                loss = criterion(outputs, label.long())
                tlosses.append(loss.detach().item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                batch_count += 1

                if (batch_count) % args.step_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        vlosses, vaccus, vprecs, vrecas, vf1 = [], [], [], [], []
                        for vi, vsfeatures in enumerate(valid_loader):
                            kmer, signal, label = vsfeatures
                            if use_cuda:
                                kmer = kmer.cuda()
                                signal = signal.cuda()
                                label = label.cuda()
                            voutputs, vlogits = model(kmer, signal)

                            vloss = criterion(voutputs, label.long())

                            _, vpredicted = torch.max(vlogits.data, 1)

                            if use_cuda:
                                label = label.cpu()
                                vpredicted = vpredicted.cpu()
                            i_accuracy = metrics.accuracy_score(label.numpy(), vpredicted)
                            # i_precision = metrics.precision_score(label.numpy(), vpredicted)
                            i_recall = metrics.recall_score(label.numpy(), vpredicted)
                            f1 = metrics.f1_score(label.numpy(), vpredicted)

                            vaccus.append(i_accuracy)
                            # vprecs.append(i_precision)
                            vrecas.append(i_recall)
                            vf1.append(f1)
                            vlosses.append(vloss.item())

                        if np.mean(vf1) > curr_best_f1_epoch:
                            curr_best_f1_epoch = np.mean(vf1)
                            torch.save(model.state_dict(),
                                       model_dir + args.model_type + '.b{}_s{}_epoch{}_{}.ckpt'.format(args.seq_len, args.signal_len, epoch + 1, batch_count))
                            if curr_best_f1_epoch > curr_best_f1:
                                curr_best_f1 = curr_best_f1_epoch
                                no_best_model = False

                        time_cost = time.time() - start
                        print('Epoch [{}/{}], Step [{}], TrainLoss: {:.4f}; '
                              'ValidLoss: {:.4f}, '
                              'Accuracy: {:.4f}, Recall: {:.4f}, F1: {:.4f},'
                              'curr_epoch_best_f1: {:.4f}; Time: {:.2f}s'
                              .format(epoch + 1, args.max_epoch_num, batch_count, np.mean(tlosses),
                                      np.mean(vlosses), np.mean(vaccus), np.mean(vrecas), np.mean(vf1),
                                      curr_best_f1_epoch, time_cost))
                        tlosses = []
                        start = time.time()
                        sys.stdout.flush()
                    model.train()

        model.eval()
        with torch.no_grad():
            vlosses, vaccus, vprecs, vrecas, vf1 = [], [], [], [], []
            for vi, vsfeatures in enumerate(valid_loader):
                kmer, signal, label = vsfeatures
                if use_cuda:
                    kmer = kmer.cuda()
                    signal = signal.cuda()
                    label = label.cuda()
                voutputs, vlogits = model(kmer, signal)

                vloss = criterion(voutputs, label.long())

                _, vpredicted = torch.max(vlogits.data, 1)

                if use_cuda:
                    label = label.cpu()
                    vpredicted = vpredicted.cpu()
                i_accuracy = metrics.accuracy_score(label.numpy(), vpredicted)
                # i_precision = metrics.precision_score(label.numpy(), vpredicted, zero_division=0)
                f1 = metrics.f1_score(label.numpy(), vpredicted)
                i_recall = metrics.recall_score(label.numpy(), vpredicted)

                vaccus.append(i_accuracy)
                # vprecs.append(i_precision)
                vrecas.append(i_recall)
                vf1.append(f1)
                vlosses.append(vloss.item())

            torch.save(model.state_dict(),
                       model_dir + args.model_type + '.b{}_s{}_epoch{}_{}.ckpt'.format(args.seq_len, args.signal_len, epoch + 1, batch_count))

            if np.mean(vf1) > curr_best_f1_epoch:
                curr_best_f1_epoch = np.mean(vf1)
                if curr_best_f1_epoch > curr_best_f1:
                    curr_best_f1 = curr_best_f1_epoch
                    no_best_model = False

            time_cost = time.time() - start
            print('Epoch [{}/{}], Step [{}], TrainLoss: {:.4f}; '
                  'ValidLoss: {:.4f}, '
                  'Accuracy: {:.4f}, Recall: {:.4f}, F1: {:.4f},'
                  'curr_epoch_best_f1: {:.4f}; Time: {:.2f}s'
                  .format(epoch + 1, args.max_epoch_num, batch_count, np.mean(tlosses),
                          np.mean(vlosses), np.mean(vaccus), np.mean(vrecas), np.mean(vf1),
                          curr_best_f1_epoch, time_cost))
            tlosses = []
            start = time.time()
            sys.stdout.flush()
        model.train()

        scheduler.step()
        if no_best_model and epoch >= args.min_epoch_num - 1:
            print("early stop!")
            break

    endtime = time.time()
    clear_linecache()
    print("[main] train costs {} seconds, best accuracy: {}".format(endtime - total_start, curr_best_f1))


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--train_file1', type=str, required=True)
    parser.add_argument('--train_file2', type=str, required=False)
    parser.add_argument('--train_file3', type=str, required=False)
    parser.add_argument('--train_file4', type=str, required=False)

    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)

    # model input
    parser.add_argument('--model_type', type=str, default="both_bilstm",
                        choices=["both_bilstm", "seq_bilstm", "signal_bilstm"],
                        required=False,
                        help="type of model to use, 'both_bilstm', 'seq_bilstm' or 'signal_bilstm', "
                             "'both_bilstm' means to use both seq and signal bilstm, default: both_bilstm")
    parser.add_argument('--seq_len', type=int, default=13, required=False,
                        help="len of kmer. default 13")
    parser.add_argument('--signal_len', type=int, default=15, required=False,
                        help="the number of signals of one base to be used in deepsignal_plant, default 15")

    # model param
    parser.add_argument('--layernum1', type=int, default=3,
                        required=False, help="lstm layer num for combined feature, default 3")
    parser.add_argument('--layernum2', type=int, default=2,
                        required=False, help="lstm layer num for seq feature (and for signal feature too), default 1")
    parser.add_argument('--class_num', type=int, default=2, required=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5, required=False)
    parser.add_argument('--n_vocab', type=int, default=16, required=False,
                        help="base_seq vocab_size (15 base kinds from iupac)")
    parser.add_argument('--n_embed', type=int, default=4, required=False,
                        help="base_seq embedding_size")
    parser.add_argument('--is_base', type=str, default="yes", required=False,
                        help="is using base features in seq model, default yes")
    parser.add_argument('--is_signallen', type=str, default="yes", required=False,
                        help="is using signal length feature of each base in seq model, default yes")
    parser.add_argument('--is_trace', type=str, default="yes", required=False,
                        help="is using trace (base prob) feature of each base in seq model, default yes")

    # BiLSTM model param
    parser.add_argument('--hid_rnn', type=int, default=256, required=False,
                        help="BiLSTM hidden_size for combined feature")

    # model training
    parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "RMSprop", "SGD"],
                        required=False, help="type of optimizer to use, 'Adam' or 'SGD' or 'RMSprop', default Adam")
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--lr', type=float, default=0.001, required=False)
    parser.add_argument("--max_epoch_num", action="store", default=10, type=int,
                        required=False, help="max epoch num, default 10")
    parser.add_argument("--min_epoch_num", action="store", default=5, type=int,
                        required=False, help="min epoch num, default 5")
    parser.add_argument('--step_interval', type=int, default=5000, required=False)

    parser.add_argument('--pos_weight', type=float, default=1.0, required=False)
    # parser.add_argument('--seed', type=int, default=1234,
    #                     help='random seed')

    # else
    parser.add_argument('--tmpdir', type=str, default="/tmp", required=False)
    parser.add_argument('--pretrained_model', type=str, required=False)

    args = parser.parse_args()

    print("[main] start..")
    total_start = time.time()

    display_args(args)

    fix_seeds()
    train(args)

    endtime = time.time()
    print("[main] costs {} seconds".format(endtime - total_start))


if __name__ == '__main__':
    main()
