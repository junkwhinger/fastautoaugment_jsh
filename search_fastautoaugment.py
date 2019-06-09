import os
import argparse
import pickle
import json
import logging
import glob
from collections import defaultdict
from itertools import combinations

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as TF
import tensorboardX

from hyperopt import fmin, hp, tpe, STATUS_OK, Trials

import utils
from augmentations import *
from train import *
from model import net, data_loader


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
        policy_dict = json.load(open("experiments/fastautoaugment/optimal_policy.json", "rb"))
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


# as in FAA
def fetch_aug_pool():
    """
    Method that returns augmentation functions and their corresponding value ranges
    :return: augmentation list
    """
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
    return augmentation_list_to_explore



def aug_list_to_hp_format(aug_dict):
    """
    Method that translates augmentation dict into HyperOpt format
    Our search space is similar to previous methods except that we use both continuous values of
    probability p and magnitude λ at [0, 1] ...

    :param aug_dict: augmentation dictionary
    :return: list of augmentations with HyperOpt variables
    """
    hp_format = []
    for aug_name, aug_range in aug_dict.items():

        if aug_range is not None:
            hp_format.append({
                aug_name: {aug_name + "_p": hp.uniform(aug_name + "_p", 0, 1),
                           aug_name + "_v": hp.uniform(aug_name + "_v", 0, 1)}
            })

        else:
            hp_format.append({
                aug_name: {aug_name + "_p": hp.uniform(aug_name + "_p", 0, 1),
                           aug_name + "_v": None
                           }
            })

    return hp_format

def convert_to_sub_policies(augmentation_list, num_operations=2):
    """
    Method that generates combinations of augmentations
    :param augmentation_list: augmentation list
    :param num_operations: number of operations to connect back to back
    :return: list of sub-policies
    """
    comb_list = combinations(augmentation_list, num_operations)
    sub_policies_list = []
    for comb in comb_list:
        sub_policies_list.append(comb)
    return sub_policies_list

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

    fn_1 = transform_fn.transforms[0].__class__.__name__
    fn_2 = transform_fn.transforms[1].__class__.__name__

    for idx, (input_batch, target_batch) in enumerate(loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        if header == "k0_t0":
            imgs_to_show = input_batch[:16]
            writer.add_image("{}__{}".format(fn_1, fn_2), make_grid(imgs_to_show, nrow=4, normalize=True))

        with torch.no_grad():
            output_batch = network.forward(input_batch)

            loss_batch = loss_function(output_batch, target_batch)
            _, prediction_batch = output_batch.max(1)
            correct_batch = prediction_batch.eq(target_batch).sum().item()

        metric_watcher.update(loss_batch.item() * input_batch.size(0),
                              correct_batch,
                              input_batch.size(0))

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

    transform_bayesian = CustomCompose(base=[
        # TF.RandomHorizontalFlip(0.5),
        # TF.RandomCrop(32, padding=4),
        TF.ToTensor(),
        TF.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_bayesian.build(space)

    loss_function = nn.CrossEntropyLoss()
    _, avg_loss, error = evaluate_error(net_m, loader_a, loss_function, header, device, transform_bayesian, writer)

    transform_bayesian.reset()

    return error


def bayesian_optimization(dataset, model, batch_size, device, policies_to_search, max_iter, header, txwriter):
    """
    Method to run bayesian optimization
    :param dataset: dataset
    :param model: network to evaluate
    :param batch_size: batch_size for evaluation
    :param device: cpu or cuda
    :param policies_to_search: list of sub-policy combinations
    :param max_iter: search depth for optimization
    :param header: header to log
    :param txwriter: tensorboard writer
    :return:
    """

    def f(space):
        val_error = hyperopt_train_test(space)
        return {'loss': val_error, 'status': STATUS_OK}

    search_space = defaultdict()
    search_space['sub_policy'] = hp.choice('sub_policy', policies_to_search)

    search_space['device'] = device
    search_space['dataset'] = dataset
    search_space['model'] = model
    search_space['batch_size'] = batch_size
    search_space['header'] = header
    search_space['writer'] = txwriter

    trials = Trials()
    best = fmin(f,
                search_space,
                algo=tpe.suggest,
                max_evals=max_iter,
                trials=trials)

    return best, trials


def decipher_trial(trial):
    """
    Method that extract sub-policies and their losses from Trial records
    :param trial: trials recorded during Bayesian Optimization
    :return: list of errors, list of sub-policies
    """
    val_error_list = [t['result']['loss'] for t in trial.trials]
    trial_records = [t['misc']['vals'] for t in trial.trials]

    sub_policy_list = []
    for record in trial_records:
        valid_record = defaultdict()
        for k, v in record.items():
            if v != [] and k != 'sub_policy':
                valid_record[k] = v[0]
        op_names = (set(vr_key.split("_")[0] for vr_key in valid_record.keys()))

        sub_policy = {}
        for op_name in op_names:
            if op_name + "_v" in valid_record:
                sub_policy[op_name] = [valid_record[op_name + "_p"], valid_record[op_name + "_v"]]
            else:
                sub_policy[op_name] = [valid_record[op_name + "_p"], "None"]

        sub_policy_list.append(sub_policy)

    return val_error_list, sub_policy_list

def extract_best_policies(search_results_folder, cv_folds, search_width, topN):
    """
    Method that returns the best augmentation policies from deciphered trials
    :param search_results_folder: where the trials are saved
    :param cv_folds: number of splits
    :param search_width: search width
    :param topN: top N policies to select at each fold
    :return: the final set of best policies
    """

    total_best_policies = {}

    for k_idx in range(cv_folds):

        byT_error = []
        byT_policies = []

        for t_idx in range(search_width):
            trials = pickle.load(
                open(os.path.join(search_results_folder, "k{}_t{}_trials.pkl".format(k_idx, t_idx)), "rb"))
            val_error_list, sub_policy_list = decipher_trial(trials)

            byT_error.extend(val_error_list)
            byT_policies.extend(sub_policy_list)

        topN_results = sorted(zip(byT_policies, byT_error), key=lambda x: x[1])[:topN]
        topN_error = []
        for idx, entry in enumerate(topN_results):
            total_best_policies[k_idx * 10 + idx] = (entry[0])
            topN_error.append(entry[1])
        logging.info(" - Average Error of the searched policies: {:.3f} at cv {}".format(np.mean(topN_error) * 100, k_idx))

    return total_best_policies


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
    # run hyperparameter search
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

            # load augmentation space to search
            aug_space_to_explore = fetch_aug_pool()
            aug_space_in_hp = aug_list_to_hp_format(aug_space_to_explore)
            sub_policies = convert_to_sub_policies(aug_space_in_hp, hparams.number_of_ops)

            # Bayesian optimization
            for t_idx in range(hparams.search_width):
                logging.info("- k:{} t: {}".format(k_idx, t_idx))

                best, trials = bayesian_optimization(dataset_a,
                                                     model_m,
                                                     hparams.batch_size * 16,
                                                     device,
                                                     sub_policies,
                                                     hparams.search_depth,
                                                     header="k{}_t{}".format(k_idx, t_idx),
                                                     txwriter=writer)

                trial_path = os.path.join(args.model_dir, "k{}_t{}_trials.pkl".format(k_idx, t_idx))
                pickle.dump(trials, open(trial_path, "wb"))


    # -- 4. Merge
    # pick N per fold
    logging.info("running -- Merge")
    total_best_policies = extract_best_policies(args.model_dir,
                          cv_folds=hparams.cv_folds,
                          search_width=hparams.search_width,
                          topN=hparams.topN)

    policy_path = os.path.join(args.model_dir, "optimal_policy.json")

    # save the optimal policy as a json file
    json.dump(total_best_policies, open(policy_path, 'w'), indent=4, sort_keys=False)

    logging.info("Optimal Policy saved. Done.")