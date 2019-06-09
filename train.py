import os
import argparse
import logging

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import tensorboardX

# custom python files
import utils
import search_fastautoaugment as SF
from model import net, data_loader

# GradualWarmupScheduler from https://github.com/ildoonet/pytorch-gradual-warmup-lr
from warmup_scheduler import GradualWarmupScheduler



def train_epoch(network, loader, loss_function, optimizer, header, device, writer=None):
    """Method that trains network with train loader"""
    network.train()

    metric_watcher = utils.RunningAverage()

    for idx, (input_batch, target_batch) in enumerate(loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        if writer is not None:
            if idx == 0:
                imgs_to_show = input_batch[:16]
                writer.add_image("augmented_inputs".format(), make_grid(imgs_to_show, nrow=4, normalize=True))

        output_batch = network.forward(input_batch)

        loss_batch = loss_function(output_batch, target_batch)
        _, prediction_batch = output_batch.max(1)
        correct_batch = prediction_batch.eq(target_batch).sum().item()

        # update network
        optimizer.zero_grad()
        nn.utils.clip_grad_norm_(network.parameters(), 5)
        loss_batch.backward()
        optimizer.step()

        metric_watcher.update(loss_batch * input_batch.size(0),
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

def evaluate_epoch(network, loader, loss_function, header, device):
    """Method that evaluates network with valid or test loader
       This method is also used for fastautoaugment
    """
    network.eval()

    metric_watcher = utils.RunningAverage()

    for idx, (input_batch, target_batch) in enumerate(loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

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




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch FastAutoAugment Implementation by Junsik Hwang")
    parser.add_argument("--model_dir", default="experiments/fastautoaugment")
    parser.add_argument("--eval_only", action='store_true')
    args = parser.parse_args()

    # load predefined hyper parameter set
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

    logging.info("\n\n--------------------------------")
    for hparam, value in hparams.__dict__.items():
        logging.info("{}: {}".format(hparam, value))

    # Prepare image augmentation policies (baseline, fastautoaugment)
    transform_train, transform_valid = SF.prepare_transform_fn(hparams.experiment_title)

    # prepare dataset_train, dataset_valid, dataset_test
    logging.info("-- loading datasets")
    dataset_train, dataset_valid, dataset_test, _ = data_loader.fetch_datasets(transform_train, transform_valid)

    # prepare loaders
    loader_train = DataLoader(dataset_train, batch_size=hparams.batch_size, shuffle=True, num_workers=2)
    loader_valid = DataLoader(dataset_valid, batch_size=hparams.batch_size, shuffle=False, num_workers=2)
    loader_test = DataLoader(dataset_test, batch_size=hparams.batch_size, shuffle=False, num_workers=2)

    # prepare model
    model = net.WideResNet(hparams.WRN_depth,
                           hparams.WRN_widen_factor,
                           hparams.WRN_dropout_rate,
                           hparams.WRN_num_classes)
    model.to(device)

    # prepare loss_fn, optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=hparams.SGD_lr,
                          momentum=hparams.SGD_momentum,
                          weight_decay=hparams.SGD_weight_decay,
                          nesterov=hparams.SGD_nesterov
                          )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=hparams.num_epochs,
        eta_min=0.0
    )

    # as in the Official FastAutoAugment implementation
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=hparams.LRS_multiplier,
        total_epoch=hparams.LRS_epoch,
        after_scheduler=scheduler
    )

    writer = tensorboardX.SummaryWriter(args.model_dir)

    # if not eval_only, run training
    if not args.eval_only:

        logging.info("-- start_training for {} epochs".format(hparams.num_epochs))

        for epoch in range(1, hparams.num_epochs + 1):
            scheduler.step()
            logging.info("Epoch - {:03d} learning_rate: {}".format(epoch, optimizer.param_groups[0]['lr']))
            model, train_loss, train_error = train_epoch(model, loader_train, loss_fn, optimizer,
                                                         header="  TRAIN", device=device, writer=writer)
            _, valid_loss, valid_error = evaluate_epoch(model, loader_valid, loss_fn, header="  VALID", device=device)
            logging.info("\n")

            writer.add_scalars('data/losses', {'train_loss': train_loss,
                                               'valid_loss': valid_loss},
                               epoch)
            writer.add_scalars('data/error', {'train_error': train_error,
                                              'valid_error': valid_error},
                               epoch)

            # save model at epoch 200 (assuming the best model is obtained at epoch 200
            if epoch == hparams.num_epochs:
                model_path = os.path.join(args.model_dir, "best_model.torch")
                torch.save(model.state_dict(), model_path)

    logging.info("-- start testing on test set")
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.torch"), map_location=device))
    evaluate_epoch(model, loader_test, loss_fn, header="TEST", device=device)




