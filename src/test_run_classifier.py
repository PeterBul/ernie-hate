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
            full_train.append((train_arr[-3] + '/conv')
        else:
            full_train.append(train_arr[-2])
        train_set = full_train.join('-')
        test_arr = args.test_set.split('/')
        if test_arr[-2] == 'conv':
            test_set = test_arr[-3] + '/conv'
        else:
            test_set = train_arr[-2]
    return train_set, test_set

