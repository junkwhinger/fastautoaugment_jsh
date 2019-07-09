import os
import argparse
import pickle
import json
import logging
import glob
from collections import defaultdict
from itertools import combinations
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import transforms as vtransforms
from torchvision.utils import make_grid
import torchvision.transforms as TF
import tensorboardX

from hyperopt import fmin, hp, tpe, STATUS_OK, Trials

import utils
from augmentations import *
from train import *
from model import net, data_loader

augmentation_list_to_explore = {
        'ShearX': [-0.3, 0.3],
        'ShearY': [-0.3, 0.3],
        'TranslateX': [-0.45, 0.45],
        'TranslateY': [-0.45, 0.45],
        'Rotate': [-30, 30],
        'AutoContrast': None,
        'Invert': None,
        'Equalize': None,
        'Solarize': [0, 256],
        'Posterize': [4, 8],
        'Contrast': [0.1, 1.9],
        'Color': [0.1, 1.9],
        'Brightness': [0.1, 1.9],
        'Sharpness': [0.1, 1.9],
        'Cutout': [0.0, 0.2]
}


def prepare_transform_fn(experiment_title):
    """
    Method that returns a set of augmentations for experiments
    :param experiment_title: "baseline" or "fastautoaugment"
    :return: transform functions for train and validation dataset
    """

    base = TF.Compose([
        TF.RandomHorizontalFlip(0.5),
        TF.RandomCrop(32, padding=4),
        TF.ToTensor(),
        TF.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_valid = TF.Compose([
        TF.ToTensor(),
        TF.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if experiment_title == 'baseline':
        transform_train = base

    elif experiment_title == 'fastautoaugment':
        # load pretrained recipes
        policy_dict = {'policy': pickle.load(open("experiments/fastautoaugment/optimal_policy.pkl", "rb"))}
        base.transforms.insert(0, FAAaugmentation(policy_dict))
        transform_train = base
    else:
        raise NotImplementedError

    return transform_train, transform_valid


def generate_cv_sets(dataset, labels, cv_folds, seed=0):
    """
    Method that generates cross validation sets
    :param dataset: dataset to split
    :param labels: labels to stratify the split
    :param cv_folds: number of splits
    :param seed: random seed
    :return: splitted indicies for D_Ms and D_As
    """

    splitter = StratifiedKFold(n_splits=cv_folds, random_state=seed)
    cv_splitter = splitter.split(np.arange(len(dataset)), labels)

    cv_sets = []
    for _ in range(cv_folds):
        cv_sets.append(next(cv_splitter))

    return cv_sets


def split_dataset_train(dataset, cvs):
    """
    Method that splits D_Train into D_Ms and D_As
    :param dataset: train_dataset
    :param cvs: cross-validation sets of indices
    :return: list of D_Ms and D_As
    """
    ds_m_list = []
    ds_a_list = []
    for cv in cvs:
        m_indices, a_indices = cv

        ds_m = Subset(dataset, m_indices)
        ds_a = Subset(dataset, a_indices)

        ds_m_list.append(ds_m)
        ds_a_list.append(ds_a)

    return ds_m_list, ds_a_list


def update_tranform_fn(dataset, transform_to_update):
    """
    Method that overwrites its predefined transform function with a new one
    :param dataset: target dataset
    :param transform_to_update: new transform function
    :return: dataset with a new transform function
    """
    st = dataset
    while hasattr(st, 'dataset'):
        st = st.dataset
    st.transform = transform_to_update
    return dataset


def evaluate_error(network, loader, loss_function, header, device, transform_fn=None, writer=None):
    """
    Method that evaluates network with valid or test loader
    :param network: model to evaluate
    :param loader: dataloader to load images and targets
    :param loss_function: loss function
    :param header: phase to log
    :param device: cpu or cuda
    :param transform_fn: new transform function to update
    :param writer: tensorboardX writer
    :return: model, average_loss, error
    """
    if transform_fn:
        loader = update_tranform_fn(loader, transform_fn)

    network.eval()

    metric_watcher = utils.RunningAverage()

    for idx, (input_batch, target_batch) in enumerate(loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        if header == "k0_t0":
            imgs_to_show = input_batch[:16]
            writer.add_image("Augmentation", make_grid(imgs_to_show, nrow=4, normalize=True))

        with torch.no_grad():
            output_batch = network.forward(input_batch)

            loss_batch = loss_function(output_batch, target_batch)
            _, prediction_batch = output_batch.max(1)
            correct_batch = prediction_batch.eq(target_batch).sum().item()

        metric_watcher.update(loss_batch.item() * input_batch.size(0),
                              correct_batch,
                              input_batch.size(0))

        #### DEBUGGGG
        # break

    metric_watcher.calculate()
    avg_loss, accuracy, error, data_points = metric_watcher()

    logging.info("{}: \tloss: {:.5f} \taccuracy: {:.1f}% \terror: {:.1f}% \tdata: {}".format(header,
                                                                                             avg_loss,
                                                                                             accuracy * 100,
                                                                                             error * 100,
                                                                                             data_points))
    return network, avg_loss, error


def hyperopt_train_test(space):
    """
    HyperOpt eval function
    :param space: search space
    :return: error to minimise
    """

    writer = space['writer']
    header = space['header']
    device = space['device']
    loader_a = DataLoader(space['dataset'],
                          batch_size=space['batch_size'],
                          shuffle=False,
                          num_workers=2)

    net_m = space['model'].to(device)

    del space['writer']
    del space['header']
    del space['device']
    del space['dataset']
    del space['model']
    del space['batch_size']

    transform_bayesian = vtransforms.Compose([
        TF.ToTensor(),
        TF.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_bayesian.transforms.insert(0, FAAaugmentation(space))

    loss_function = nn.CrossEntropyLoss()
    _, avg_loss, error = evaluate_error(net_m, loader_a, loss_function, header, device, transform_bayesian, writer)

    return error


def bayesian_optimization(dataset, model, args, device, aug_space, header, txwriter):
    """
    Method to run bayesian optimization
    :param dataset: dataset
    :param model: network to evaluate
    :param device: cpu or cuda
    :param aug_space: list of augmentations to explore\
    :param header: header to log
    :param txwriter: tensorboard writer
    :return:
    """

    def f(space):
        val_error = hyperopt_train_test(space)
        return {'loss': val_error, 'status': STATUS_OK}

    sub_policies_pool = list(aug_space.keys())
    search_space = defaultdict()

    # compose policy search space
    policy_to_eval = []
    nb_sub_policies = args.number_of_sub_policies
    nb_operations = args.number_of_ops
    for sp_idx in range(nb_sub_policies):
        sp = {}

        op_list = []
        for op_idx in range(nb_operations):
            op_element = []
            op_element.append(hp.choice("sp_{}_{}".format(sp_idx, op_idx), sub_policies_pool))
            op_element.append(hp.uniform("sp_{}_{}_p".format(sp_idx, op_idx), 0, 1))
            op_element.append(hp.uniform("sp_{}_{}_v".format(sp_idx, op_idx), 0, 1))
            op_list.append(op_element)
        sp['sp_{}'.format(sp_idx)] = op_list
        policy_to_eval.append(sp)

    search_space['policy'] = policy_to_eval
    search_space['device'] = device
    search_space['dataset'] = dataset
    search_space['model'] = model
    search_space['batch_size'] = args.batch_size * 32
    search_space['header'] = header
    search_space['writer'] = txwriter

    trials = Trials()
    best = fmin(f,
                search_space,
                algo=tpe.suggest,
                max_evals=args.search_depth,
                trials=trials)

    return best, trials


def parse_trial_records(trial_records, nb_sub_policies, nb_operations, aug_list):
    """
    Method that parse hyperopt trial records
    :param trial_records: trial_records from hyperopt
    :param nb_sub_policies: number of sub_policies in a policy
    :param nb_operations: number of consecutive operations in a sub_policy
    :param aug_list: augmentation list to explore
    :return: a list of parsed policies
    """
    aug_fns = list(aug_list.keys())

    parsed_policy_list = []
    for tr in trial_records:
        parsed_policy = []
        for sp_idx in range(nb_sub_policies):
            parsed_sub_policy = []
            for op_idx in range(nb_operations):
                parsed_operation = []
                op_num = tr['sp_{}_{}'.format(sp_idx, op_idx)][0]
                op_name = aug_fns[op_num]

                op_p = tr['sp_{}_{}_p'.format(sp_idx, op_idx)][0]
                op_v = tr['sp_{}_{}_v'.format(sp_idx, op_idx)][0]

                parsed_operation.append(op_name)
                parsed_operation.append(op_p)
                parsed_operation.append(op_v)

                parsed_sub_policy.append(parsed_operation)
            parsed_policy.append(parsed_sub_policy)
        parsed_policy_list.append(parsed_policy)
    return parsed_policy_list


def extract_best_policies(search_results_folder, cv_folds, search_width, nb_sub_policies, nb_operations, topN, aug_list):
    """
    Method that returns the best augmentation policies from deciphered trials
    :param search_results_folder: where the trials are saved
    :param cv_folds: number of splits
    :param search_width: search width
    :param nb_sub_policies: number of sub_policies in a policy
    :param nb_operations: number of consecutive operations in a sub_policy
    :param topN: top N policies to select at each fold
    :return: the final set of best policies
    """

    T_star = []

    for k_idx in range(cv_folds):

        T_star_k_idx = []

        for t_idx in range(search_width):
            trials = pickle.load(
                open(os.path.join(search_results_folder, "k{}_t{}_trials.pkl".format(k_idx, t_idx)), "rb"))

            val_error_list = [t['result']['loss'] for t in trials.trials]
            trial_records = [t['misc']['vals'] for t in trials.trials]

            parsed_policies = parse_trial_records(trial_records, nb_sub_policies, nb_operations, aug_list)
            zipped = zip(parsed_policies, val_error_list)
            top_results = sorted(zipped, key=lambda t: t[1])[:topN]
            top_policies = [res[0] for res in top_results]
            T_star_k_idx.append(top_policies)

        T_star.append(T_star_k_idx)

    flat_K = [item for sublist in T_star for item in sublist]
    flat_T = [item for sublist in flat_K for item in sublist]
    flat_N = [item for sublist in flat_T for item in sublist]

    return flat_N


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FastAutoAugment policy search by Junsik Hwang")
    parser.add_argument("--model_dir", default="fastautoaugment")
    parser.add_argument("--train_mode", action='store_true')
    parser.add_argument("--bayesian_mode", action='store_true')
    
    args = parser.parse_args()

    # load predefined hyper parameter set
    # train theta on D_Ms following the small models of AutoAugment
    # The use of a small Wide-ResNet is
    # for computational efficiency as each child model is trained
    # from scratch to compute the gradient update for the controller. We use a weight decay of 10−4
    # , learning rate of 0.01, and a cosine learning decay with one annealing cycle.
    params_path = os.path.join(args.model_dir, "params.json")
    hparams = utils.Params(params_path)

    # use GPU if available
    hparams.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if hparams.cuda else "cpu")

    # setting random seed
    torch.manual_seed(0)
    if hparams.cuda: torch.cuda.manual_seed(0)

    # logger, ready
    log_path = os.path.join(args.model_dir, "train.log")
    utils.set_logger(log_path)

    # load predefined hyper parameter set
    params_path = os.path.join(args.model_dir, "params.json")
    hparams = utils.Params(params_path)

    logging.info("\n\n--------------------------------")
    for hparam, value in hparams.__dict__.items():
        logging.info("{}: {}".format(hparam, value))

    # define transform_fn
    # The authors said they did not use data augmentation,
    # but I interpreted it as not adding extra data augmentation for this training process.
    # The purpose of the search is to find extra augmentation techniques on top of the baseline preprocessing.
    # > Next, we train model parameter θ on DM from scratch without data augmentation.
    transform_train = TF.Compose([
        # TF.RandomHorizontalFlip(0.5),
        # TF.RandomCrop(32, padding=4),
        TF.ToTensor(),
        TF.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    # 3.3 Implementation
    # -- 1. Shuffle
    logging.info("running -- Shuffling dataset_train")

    # load dataset_train
    dataset_train, dataset_valid, dataset_test, labels_train = data_loader.fetch_datasets(transform_train, None)

    # generate cv indices
    cv_sets = generate_cv_sets(dataset_train, labels_train, cv_folds=5, seed=0)

    # split dataset_train into dataset_m and dataset_a
    dataset_m_list, dataset_a_list = split_dataset_train(dataset_train, cv_sets)

    if args.train_mode:
        # -- 2. Train
        logging.info("running -- Train models on cv sets")
        for k_idx, dataset_m in enumerate(dataset_m_list):

            # get loader
            m_loader = DataLoader(dataset_m, batch_size=hparams.batch_size, shuffle=True, num_workers=2)

            # get model
            model = net.WideResNet(hparams.WRN_depth,
                                 hparams.WRN_widen_factor,
                                 hparams.WRN_dropout_rate,
                                 hparams.WRN_num_classes
                                 )
            model.apply(net.conv_init)
            model.to(device)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(),
                                  lr=hparams.SGD_lr,
                                  momentum=hparams.SGD_momentum,
                                  weight_decay=hparams.SGD_weight_decay,
                                  nesterov=hparams.SGD_nesterov)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.num_epochs,
                eta_min=0.0
            )

            # run training for the predefined epochs
            for epoch in range(1, hparams.num_epochs + 1):
                scheduler.step()
                logging.info("Epoch - {:03d} learning_rate: {}".format(epoch, optimizer.param_groups[0]['lr']))

                model, train_loss, train_error = train_epoch(model, m_loader, loss_fn, optimizer, header="  TRAIN-k{}".format(k_idx), device=device)

            # evaluate on dataset_valid
            loader_valid = DataLoader(dataset_valid, batch_size=hparams.batch_size, shuffle=False, num_workers=2)
            _, valid_loss, valid_error = evaluate_epoch(model, loader_valid, loss_fn, header="  VALID-k{}".format(k_idx), device=device)

            model_path = os.path.join(args.model_dir, "model_k_{}.torch".format(k_idx))
            torch.save(model.state_dict(), model_path)

            logging.info("\n")

    # load models
    logging.info("loading trained models on D_Ms")
    model_paths = glob.glob(os.path.join(args.model_dir, "*.torch"))

    # -- Explore-and-Exploit
    logging.info("running -- Explore-and-Exploit")
    # run hyper-parameter search
    if args.bayesian_mode:
        writer = tensorboardX.SummaryWriter(args.model_dir)

        # for each model that has been fitted to D_M_k
        for k_idx in range(hparams.cv_folds):

            # load model
            model_m = net.WideResNet(hparams.WRN_depth,
                                     hparams.WRN_widen_factor,
                                     hparams.WRN_dropout_rate,
                                     hparams.WRN_num_classes)
            weight_path = os.path.join(args.model_dir, "model_k_{}.torch".format(k_idx))
            model_m.load_state_dict(torch.load(weight_path, map_location=device))

            # prepare D_A
            dataset_a = dataset_a_list[k_idx]

            # Bayesian optimization
            for t_idx in range(hparams.search_width):
                logging.info("- k:{} t: {}".format(k_idx, t_idx))

                best, trials = bayesian_optimization(dataset_a,
                                                     model_m,
                                                     hparams,
                                                     device,
                                                     augmentation_list_to_explore,
                                                     header="k{}_t{}".format(k_idx, t_idx),
                                                     txwriter=writer)

                trial_path = os.path.join(args.model_dir, "k{}_t{}_trials.pkl".format(k_idx, t_idx))
                pickle.dump(trials, open(trial_path, "wb"))


    # -- 4. Merge
    # pick N per fold
    logging.info("running -- Merge")
    optimal_policies = extract_best_policies(args.model_dir,
                                             cv_folds=hparams.cv_folds,
                                             search_width=hparams.search_width,
                                             nb_sub_policies=hparams.number_of_sub_policies,
                                             nb_operations=hparams.number_of_ops,
                                             topN=hparams.topN,
                                             aug_list=augmentation_list_to_explore)

    policy_path = os.path.join(args.model_dir, "optimal_policy.pkl")

    # save the optimal policy as a json file
    pickle.dump(optimal_policies, open(policy_path, "wb"))

    logging.info("Optimal Policy saved. Done.")