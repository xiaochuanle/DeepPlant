import sys

import numpy as np
import random
import os


def balance(path, output_path, k=13, bala_size=13):
    max_num = 1000000000

    loc = int(k / 2)
    input = np.load(path)
    pos_index_map = {}
    neg_index_map = {}
    pos_count = 0
    for i in range(input.shape[0]):
        if i % 1000000 == 0:
            print(i, 'samples processed, total', input.shape[0])
        # if input[i, loc] != 1.0:
        #     continue
        key = ''
        for j in range(-int(bala_size / 2), int(bala_size / 2) + 1):
            key += int(input[i, loc + j]).__str__()
        if input[i, -1] == 1.0:
            index_list = pos_index_map.get(key)
            if index_list is None:
                index_list = [i]
                pos_index_map[key] = index_list
                pos_count += 1
            else:
                index_list.append(i)
                pos_index_map[key] = index_list
                pos_count += 1
        else:
            assert input[i, -1] == 0.0
            index_list = neg_index_map.get(key)
            if index_list is None:
                index_list = [i]
                neg_index_map[key] = index_list
            else:
                index_list.append(i)
                neg_index_map[key] = index_list

    res = np.zeros(shape=(pos_count * 2, input.shape[1]), dtype=np.float32)
    pos_loss_count = 0
    j = 0
    i = 0
    k = 0
    pos_count = 0
    neg_count = 0
    for key in pos_index_map.keys():
        if j % 1000 == 0:
            print(j, 'motif processed, total', len(pos_index_map.keys()))
        j += 1
        pos_index_list = pos_index_map.get(key)
        neg_index_list = neg_index_map.get(key)
        if neg_index_list is None:
            pos_loss_count += len(pos_index_list) if len(pos_index_list) < max_num else max_num
            continue
        if len(pos_index_list) > len(neg_index_list):
            if len(neg_index_list) > max_num:
                num = max_num
            else:
                num = len(neg_index_list)
            if num == max_num:
                pos_loss_count += 0
            elif len(pos_index_list) > max_num:
                pos_loss_count += max_num - len(neg_index_list)
            else:
                pos_loss_count += len(pos_index_list) - len(neg_index_list)
        else:
            if len(pos_index_list) > max_num:
                num = max_num
            else:
                num = len(pos_index_list)
        # add pos
        rlist = random.sample(range(len(pos_index_list)), k=num)
        for r in rlist:
            index = pos_index_list[r]
            res[k, :] = input[index, :]
            k += 1
        pos_count += len(rlist)
        # add neg
        sample_num = len(pos_index_list) if len(pos_index_list) < len(neg_index_list) else len(neg_index_list)
        if sample_num > max_num:
            sample_num = max_num
        rlist = random.sample(range(len(neg_index_list)), k=sample_num)
        for r in rlist:
            index = neg_index_list[r]
            res[k, :] = input[index, :]
            k += 1
        neg_count += len(rlist)
    res = res[0:k, :]
    input = None
    pos_index_map = None
    neg_index_map = None
    print('unbalance pos loss count:', pos_loss_count)
    print('pos total:', pos_count)
    print('neg total:', neg_count)
    print('total:', res.shape[0])
    np.save(output_path, res.astype(np.float32))


def balance_multiply():
    k = 13
    bala_size = 13
    max_num = 100000
    path = 'merge.npy'
    output_path = 'bala.npy'
    multiply = 1

    loc = int(k / 2)
    input = np.load(path)
    pos_index_map = {}
    neg_index_map = {}
    pos_count = 0
    for i in range(input.shape[0]):
        if i % 1000000 == 0:
            print(i, 'samples processed, total', input.shape[0])
        # if input[i, loc] != 1.0:
        #     continue
        key = ''
        for j in range(-int(bala_size / 2), int(bala_size / 2) + 1):
            key += int(input[i, loc + j]).__str__()
        if input[i, -1] == 1.0:
            index_list = pos_index_map.get(key)
            if index_list is None:
                index_list = [i]
                pos_index_map[key] = index_list
                pos_count += 1
            else:
                index_list.append(i)
                pos_index_map[key] = index_list
                pos_count += 1
        else:
            assert input[i, -1] == 0.0
            index_list = neg_index_map.get(key)
            if index_list is None:
                index_list = [i]
                neg_index_map[key] = index_list
            else:
                index_list.append(i)
                neg_index_map[key] = index_list

    res = np.zeros(shape=(pos_count * (multiply + 1), input.shape[1]), dtype=np.float32)
    pos_loss_count = 0
    j = 0
    i = 0
    k = 0
    pos_count = 0
    neg_count = 0
    for key in pos_index_map.keys():
        if j % 1000 == 0:
            print(j, 'motif processed, total', len(pos_index_map.keys()))
        j += 1
        pos_index_list = pos_index_map.get(key)
        neg_index_list = neg_index_map.get(key)
        if neg_index_list is None:
            pos_loss_count += len(pos_index_list) if len(pos_index_list) < max_num else max_num
            continue
        if len(pos_index_list) > len(neg_index_list):
            if len(neg_index_list) > max_num:
                num = max_num
            else:
                num = len(neg_index_list)
            if num == max_num:
                pos_loss_count += 0
            elif len(pos_index_list) > max_num:
                pos_loss_count += max_num - len(neg_index_list)
            else:
                pos_loss_count += len(pos_index_list) - len(neg_index_list)
        else:
            if len(pos_index_list) > max_num:
                num = max_num
            else:
                num = len(pos_index_list)
        # add pos
        rlist = random.sample(range(len(pos_index_list)), k=num)
        for r in rlist:
            index = pos_index_list[r]
            res[k, :] = input[index, :]
            k += 1
        pos_count += len(rlist)
        # add neg
        num = len(pos_index_list) * multiply
        sample_num = num if num < len(neg_index_list) else len(neg_index_list)
        if sample_num > max_num * multiply:
            sample_num = max_num * multiply
        rlist = random.sample(range(len(neg_index_list)), k=sample_num)
        for r in rlist:
            index = neg_index_list[r]
            res[k, :] = input[index, :]
            k += 1
        neg_count += len(rlist)
    res = res[0:k, :]

    for key in pos_index_map.keys():
        print(key)

    input = None
    pos_index_map = None
    neg_index_map = None
    print('unbalance pos loss count:', pos_loss_count)
    print('pos total:', pos_count)
    print('neg total:', neg_count)
    print('total:', res.shape[0])
    np.save(output_path, res.astype(np.float32))


def balance_low_mem(path, output_path, k=13, bala_size=13):
    max_num = 1000000000

    k = 13
    bala_size = 9
    path = 'chx/0.95_0.9_0.85_0_9bala_13bp/DS'
    path2 = 'chx/0.95_0.9_0.85_0_9bala_13bp/BM'
    path3 = 'chx/0.95_0.9_0.85_0_9bala_13bp/A9'
    output_path = 'chx/0.95_0.9_0.85_0_9bala_13bp/balance'

    entries = os.listdir(path)
    file_path_names1 = [os.path.join(path, f) for f in entries if os.path.isfile(os.path.join(path, f))]
    entries = os.listdir(path2)
    file_path_names2 = [os.path.join(path2, f) for f in entries if os.path.isfile(os.path.join(path2, f))]
    entries = os.listdir(path3)
    file_path_names3 = [os.path.join(path3, f) for f in entries if os.path.isfile(os.path.join(path3, f))]
    file_path_names = file_path_names1 + file_path_names2 + file_path_names3
    # file_path_names = file_path_names1

    random.seed(13)
    loc = int(k / 2)
    pos_index_map = {}
    neg_index_map = {}
    pos_count = 0
    file_index = 0
    for file in file_path_names:
        try:
            if file.endswith('npz'):
                input = np.load(file).get('features')
            else:
                input = np.load(file)
        except:
            print(file, 'read error!')
            sys.stdout.flush()
            continue
        for i in range(input.shape[0]):
            key = ''
            for j in range(-int(bala_size / 2), int(bala_size / 2) + 1):
                key += int(input[i, loc + j]).__str__()
            if input[i, -1] == 1.0:
                index_list = pos_index_map.get(key)
                if index_list is None:
                    index_list = [(file_index, i)]
                    pos_index_map[key] = index_list
                    pos_count += 1
                else:
                    index_list.append((file_index, i))
                    pos_index_map[key] = index_list
                    pos_count += 1
            else:
                assert input[i, -1] == 0.0
                index_list = neg_index_map.get(key)
                if index_list is None:
                    index_list = [(file_index, i)]
                    neg_index_map[key] = index_list
                else:
                    index_list.append((file_index, i))
                    neg_index_map[key] = index_list
        file_index += 1
        if file_index % 100 == 0:
            print(file_index, 'files processed, total', len(file_path_names))
            sys.stdout.flush()

    pos_loss_count = 0
    j = 0
    pos_count = 0
    neg_count = 0
    res_list = []
    for key in pos_index_map.keys():
        if j % 10000 == 0:
            print(j, 'motif processed, total', len(pos_index_map.keys()))
            sys.stdout.flush()
        j += 1
        pos_index_list = pos_index_map.get(key)
        neg_index_list = neg_index_map.get(key)
        if neg_index_list is None:
            pos_loss_count += len(pos_index_list) if len(pos_index_list) < max_num else max_num
            continue
        if len(pos_index_list) > len(neg_index_list):
            if len(neg_index_list) > max_num:
                num = max_num
            else:
                num = len(neg_index_list)
            if num == max_num:
                pos_loss_count += 0
            elif len(pos_index_list) > max_num:
                pos_loss_count += max_num - len(neg_index_list)
            else:
                pos_loss_count += len(pos_index_list) - len(neg_index_list)
        else:
            if len(pos_index_list) > max_num:
                num = max_num
            else:
                num = len(pos_index_list)
        # add pos
        rlist = random.sample(range(len(pos_index_list)), k=num)
        for r in rlist:
            index_tuple = pos_index_list[r]
            res_list.append(index_tuple)
        pos_count += len(rlist)
        # add neg
        sample_num = len(pos_index_list) if len(pos_index_list) < len(neg_index_list) else len(neg_index_list)
        if sample_num > max_num:
            sample_num = max_num
        rlist = random.sample(range(len(neg_index_list)), k=sample_num)
        for r in rlist:
            index_tuple = neg_index_list[r]
            res_list.append(index_tuple)
        neg_count += len(rlist)

    res_list = np.array(res_list)
    for file_index in range(len(file_path_names)):
        index_array = res_list[res_list[:, 0] == file_index, 1]
        file = file_path_names[file_index]
        try:
            if file.endswith('npz'):
                input = np.load(file).get('features')
            else:
                input = np.load(file)
        except:
            print(file, 'read error!')
            sys.stdout.flush()
            # not work?
            continue
        out_put = input[index_array]
        np.save(output_path + '/' + file_path_names[file_index].split('/')[-1], out_put)
        if (file_index + 1) % 100 == 0:
            print(file_index + 1, 'files saved, total', len(file_path_names))
            sys.stdout.flush()

    print('unbalance pos loss count:', pos_loss_count)
    print('pos total:', pos_count)
    print('neg total:', neg_count)
    print('total:', res_list.shape[0])
    sys.stdout.flush()

