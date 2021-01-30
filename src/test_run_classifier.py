import argparse
import json

parser = argparse.ArgumentParser()
args = parser.parse_args()

def get_train_and_test_set(args):
    if args.do_train:
        full_train = []
        if args.init_checkpoint:
            init_cp = args.init_checkpoint.split('/')
            if init_cp[2][:4] == 'conv':
                full_train.append(init_cp[1] + '/conv')
            else:
                full_train.append(init_cp[2])
        train_arr = args.train_set.split('/')
        if train_arr[-2] == 'conv':
            full_train.append((train_arr[-3] + '/conv'))
        else:
            full_train.append(train_arr[-2])
        train_set = '-'.join(full_train)
        test_arr = args.test_set.split('/')
        if test_arr[-2] == 'conv':
            test_set = test_arr[-3] + '/conv'
        else:
            test_set = test_arr[-2]
    else:
        return None, None
    return train_set, test_set


def get_test_set(args):
    arr = args.test_set.split('/')
    return '/'.join(arr[2:-1])


def write_json(args, save_path):
    #write_to_tmp(args, save_path)
    insert = {"results": save_path}
    lr_text = "2e-05"
    model_size = "base"

    j = dict()
    
    archive = {'founta/conv/isaksen': {'founta/conv/isaksen': {'base': {'1e-05': {'results': 'test-save-path.test'}}, 'large': {'1e-05': {'results': 'test-save-path.test'}, '2e-05': {'results': 'test-save-path.test'}}}}}
    test_set = get_test_set(args)
    key1 = []
    if args.init_checkpoint and 'checkpoints' not in args.init_checkpoint.split('/'):
        key1.append(args.init_checkpoint)
    if args.do_train and not (args.init_checkpoint and 'checkpoints' in args.init_checkpoint.split('/')):
        key1.append(test_set)
    key1 = '-'.join(key1)
    key2 = test_set

    #"checkpoints": args.checkpoints + '/step_' + str(steps)

    if key1 != '' and key1 in archive and key2 in archive[key1]:
        j = archive[key1][key2]

    if model_size not in j:
        j[model_size] = dict()

    j[model_size][lr_text] = insert

    if key1 not in archive:
        archive[key1] = dict()
    
    archive[key1][key2] = j
    print(archive)
    #write_archive(archive)

if __name__ == "__main__":
    args.__dict__ = {'init_checkpoint': None, 'do_train': True, 'test_set': '../data/founta/conv/isaksen/test.tsv'}
    write_json(args, "test-save-path.test")
