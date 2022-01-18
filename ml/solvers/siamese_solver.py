import gc
import logging
import sys
import time
from copy import deepcopy
from itertools import chain

import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ml.models import get_model
from ml.losses import get_loss
from ml.optimizers import get_optimizer, get_lr_policy
from ml.solvers.base_solver import Solver
from utils.device import DEVICE, put_minibatch_to_device
import numpy as np

from utils.iohandler import IOHandler


class SiameseSolver(Solver):
    def __init__(self, config, args):
        """
        Solver parent function to control the experiments.
        It contains everything for an experiment to run.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        super(SiameseSolver, self).__init__(config, args)

        # deep learn specific stuff
        self.init_epochs()
        if DEVICE == torch.device('cuda'):
            self.top_model.cuda()
            self.bottom_model.cuda()
        self.init_loss()
        self.init_optimizer()
        self.init_lr_policy()
        self.iohandler = IOHandler(args, self)
        self.iohandler.load_checkpoint()
        self.stage = 2 if self.stage2_epoch > 0 else 1

    def init_epochs(self):
        """
        Initialize the epoch number initialization(s), (can be overwritten during checkpoint load).
        """
        logging.info("Initializing the epoch number.")
        self.stage1_epoch = 0
        self.stage1_epochs = self.config.env.stage1_epochs
        self.stage2_epoch = 0
        self.stage2_epochs = self.config.env.stage2_epochs

    def init_model(self):
        """
        Initialize the model according to the config and put it on the gpu if available,
        (weights can be overwritten during checkpoint load).
        """
        logging.info("Initializing the model.")
        self.top_model = get_model(self.config.model, split='top')
        self.bottom_model = get_model(self.config.model, split='bottom')

    def init_loss(self):
        """
        Initialize the loss according to the config.
        """
        if self.phase == 'train':
            logging.info("Initializing the loss.")
            self.loss = get_loss(self.config.loss)

    def init_optimizer(self):
        """
        Initialize the optimizer according to the config, (can be overwritten during checkpoint load).
        """
        if self.phase == 'train':
            logging.info("Initializing the optimizer.")
            self.optimizer = get_optimizer(self.config.optimizer,
                                           chain(self.top_model.parameters(), self.bottom_model.parameters()))

    def init_lr_policy(self):
        """
        Initialize the learning rate policy, (can be overwritten during checkpoint load).
        """
        if self.phase == 'train':
            logging.info("Initializing lr policy.")
            self.lr_policy = get_lr_policy(self.config.lr_policy, optimizer=self.optimizer)

    def train(self):
        """
        Training all the epochs with validation after every epoch.
        Save the model if it has better performance than the previous ones.
        """
        if self.stage == 1:
            logging.info(f"Start stage 1 training")

            for self.stage1_epoch in range(self.stage1_epoch, self.stage1_epochs):
                self.epochs = self.stage1_epochs
                self.epoch = self.stage1_epoch

                logging.info(f"Start training stage 1 epoch: {self.epoch}/{self.stage1_epochs}")
                self.current_mode = 'train'
                self.run_epoch()

                logging.info(f"Start evaluating stage 1 epoch: {self.epoch}/{self.stage1_epochs}")
                self.eval()

                self.lr_policy.step(self.siamese_accuracy)
                self.iohandler.save_best_checkpoint()
                self.loader.dataset.shuffle_indices()
            self.stage = 2

        if self.stage == 2:
            self.iohandler.load_checkpoint()
            self.init_optimizer()
            for self.stage2_epoch in range(self.stage2_epoch, self.stage2_epochs):
                self.epochs = self.stage2_epochs
                self.epoch = self.stage2_epoch
                logging.info(f"Start stage 2 training")
                if self.epoch % self.config.env.update_stage2_negatives_frequency == 0:

                    # train on the full dataset to not forget the easier examples
                    self.train_loader.dataset.make_splitted_data()
                    for _ in range(self.config.env.second_stage_easy_epoch_number):
                        self.current_mode = 'train'
                        self.run_epoch()
                        self.eval()

                    logging.info(f"Start stage 2 training step 1: gathering topks")
                    self.current_mode = 'gather_topks'
                    self.run_epoch()

                    top_pairs_array = self.run_topk('top')
                    bottom_pairs_array = self.run_topk('bottom')
                    _, predicted_pairs = self.combine_topks(top_pairs_array, bottom_pairs_array)
                    self.train_loader.dataset.collect_pairs(predicted_pairs)

                # update dataloader
                logging.info(f"Start training stage 2 epoch: {self.epoch}/{self.stage2_epochs}")
                self.current_mode = 'train'
                self.run_epoch()

                logging.info(f"Start evaluating stage 2 epoch: {self.epoch}/{self.stage2_epochs}")
                self.eval()

                self.iohandler.save_best_checkpoint()

        else:
            raise ValueError(f'Wrong stage number: {self.stage}')

        return self.iohandler.get_max_metric()['accuracy']

    def eval(self):
        """
        Evaluate the model and save the predictions to a csv file during testing inference.
        """
        self.current_mode = 'test' if self.phase == 'test' else 'val'

        self.topk = self.config.env.val_topk
        self.run_epoch()
        top_pairs_array = self.run_topk('top')
        bottom_pairs_array = self.run_topk('bottom')
        pairs_array, _ = self.combine_topks(top_pairs_array, bottom_pairs_array)
        predicted_tops = self.run_bipartite_matching(pairs_array)

        if self.current_mode == 'test':
            self.test_loader.dataset.save_test_images(predicted_tops)
        else:
            if self.phase == 'train' and self.siamese_accuracy > 0.5:
                self.train_loader.dataset.positive_rate = self.siamese_accuracy
            self.iohandler.append_metric({'accuracy': self.accuracy, 'siamese_accuracy': self.siamese_accuracy})
            self.save_acc()

    def before_epoch(self):
        """
        Before every epoch set the model and the iohandler to the right mode (train or eval)
        and select the corresponding loader.
        """
        self.iohandler.reset_results()
        if self.current_mode == 'train':
            self.topk = self.config.env.train_topk
            self.top_model.train()
            self.bottom_model.train()
            self.loader = self.train_loader
            self.iohandler.train()
            self.chunk_size = self.config.env.train_chunk_size
        elif self.current_mode == 'gather_topks':
            self.topk = self.config.env.train_topk
            self.top_model.eval()
            self.bottom_model.eval()
            self.loader = self.topk_gathering_loader
            self.loader.dataset.positive_rate = 1.0
            self.iohandler.val()
            self.chunk_size = self.config.env.train_chunk_size
        elif self.current_mode == 'val':
            self.top_model.eval()
            self.bottom_model.eval()
            self.loader = self.val_loader
            self.iohandler.val()
            self.chunk_size = self.config.env.val_chunk_size
        elif self.current_mode == 'test':
            self.topk = self.config.env.val_topk
            self.top_model.eval()
            self.bottom_model.eval()
            self.loader = self.test_loader
            self.iohandler.test()
            self.chunk_size = self.config.env.test_chunk_size
        else:
            raise ValueError(f'Wrong current mode: {self.current_mode}. It should be `train` or `val`.')
        torch.cuda.empty_cache()

    def run_epoch(self):
        """
        Run a full epoch according to the current self.current_mode (train or val).
        """
        self.before_epoch()

        # set loading bar
        time.sleep(0.1)
        bar_format = '{desc}|{bar:10}|[{elapsed}<{remaining},{rate_fmt}]'
        with tqdm(range(len(self.loader)), file=sys.stdout, bar_format=bar_format, position=0, leave=True) as pbar:
            preproc_t_start = time.time()
            for idx, minibatch in enumerate(self.loader):
                minibatch = put_minibatch_to_device(minibatch)
                preproc_time = time.time() - preproc_t_start

                # train
                train_t_start = time.time()
                output, loss = self.step(minibatch)
                train_time = time.time() - train_t_start

                # save results for evaluation at the end of the epoch and calculate the running metrics
                self.iohandler.append_data(minibatch, output)
                self.iohandler.update_bar_description(pbar, idx, preproc_time, train_time, loss)

                pbar.update(1)
                preproc_t_start = time.time()

        self.after_epoch()

    def step(self, minibatch):
        """
        Make one iteration step: either a train (pred+train) or a val step (pred only).

        :param minibatch: minibatch containing the input image and the labels (labels only during `train`).
        :return: output, loss
        """
        # prediction
        if self.current_mode == 'train':
            top_output = self.top_model(minibatch['tops'])
            bottom_output = self.bottom_model(minibatch['bottoms'])

            # training step
            self.optimizer.zero_grad()
            loss = self.loss(top_output, bottom_output, minibatch['labels'])
            loss.backward()
            self.optimizer.step()

        else:
            with torch.no_grad():
                top_output = self.top_model(minibatch['tops'])
                bottom_output = self.bottom_model(minibatch['bottoms'])
            loss = 0

        return {'top_output': top_output, 'bottom_output': bottom_output}, loss

    def after_epoch(self):
        """
        After every epoch collect some garbage and evaluate the current metric.
        """
        gc.collect()
        torch.cuda.empty_cache()
        print()

    def run_topk(self, part):
        """
        Runs the model to gather the topk pairs at a given orientation, the best bottoms for all the tops, or the best tops for all the bottoms.

        :param part: orientation of the topk gathering `top` or `bottom`

        :return: pairs array containg the bottoms-tops prediction score for the topk elements
        """
        top_output = self.iohandler.results['top_output']
        bottom_output = self.iohandler.results['bottom_output']
        pairs = self.iohandler.results['pairs']

        if part == 'top':
            first = top_output.unsqueeze(1)
            second = bottom_output.unsqueeze(0)
            gt = pairs[torch.argsort(pairs[:, 0]), 1].unsqueeze(-1)
        elif part == 'bottom':
            first = bottom_output.unsqueeze(1)
            second = top_output.unsqueeze(0)
            gt = pairs[:, 0].unsqueeze(-1)
        else:
            raise ValueError(f'Wrong part: {part}')

        distance_list = torch.tensor([], device=DEVICE)
        distance_index_list = torch.tensor([], device=DEVICE)

        logging.info('Running matching for tops')
        time.sleep(0.1)
        bar_format = '{desc}|{bar:10}|[{elapsed}<{remaining},{rate_fmt}]'
        with tqdm(range(first.size(0) // self.chunk_size), file=sys.stdout, bar_format=bar_format,
                  position=0, leave=True) as pbar:
            for i in range(0, first.size(0), self.chunk_size):
                current_first = first[i:i + self.chunk_size]

                distance = torch.sum((current_first * second), dim=-1)

                top_elements, top_elements_indices = torch.topk(distance, int(self.topk), dim=-1)
                distance_list = torch.cat([distance_list, top_elements], dim=0)
                distance_index_list = torch.cat([distance_index_list, top_elements_indices], dim=0)

                pbar.update(1)

        siamese_accuracy = torch.sum(torch.any(distance_index_list == gt, dim=-1)) / top_output.size(0)

        logging.info(f'Siamese accuracy of maching for the {part} elements top{self.topk}: {siamese_accuracy * 100:.2f}%')

        distance_list = distance_list.cpu().numpy()
        distance_index_list = distance_index_list.cpu().numpy()

        pairs_array = np.zeros([pairs.size(0), pairs.size(0)])
        for i, (distance, distance_index) in enumerate(zip(distance_list, distance_index_list)):
            if part == 'top':
                pairs_array[distance_index.astype(int), i] = distance
            elif part == 'bottom':
                pairs_array[i, distance_index.astype(int)] = distance
            else:
                raise ValueError(f'Wrong part: {part}')

        logging.info(f'Mathing for {part} of top{self.topk} contains {np.sum(pairs_array>0)} elements')

        return pairs_array

    def combine_topks(self, top_pairs_array, bottom_pairs_array):
        """
        Make the intersection of the best bottoms for all the tops and the best tops for all the bottoms
        """

        logging.info('Make combined matching.')

        pairs_mask = np.logical_and(top_pairs_array > 0, bottom_pairs_array > 0)
        top_pairs_array[np.logical_not(pairs_mask)] = 0
        predicted_pairs = np.argwhere(top_pairs_array > 0)

        if self.current_mode == 'val' or self.current_mode == 'test':
            logging.info('Calculate the combined accuracy.')
            pairs = self.iohandler.results['pairs']

            summer = 0
            time.sleep(0.1)
            bar_format = '{desc}|{bar:10}|[{elapsed}<{remaining},{rate_fmt}]'
            with tqdm(range(len(np.unique(predicted_pairs[:, 0]))), file=sys.stdout, bar_format=bar_format,
                      position=0, leave=True) as pbar:
                for i in np.unique(predicted_pairs[:, 0]):
                    pairs_mask = predicted_pairs[:, 0] == i
                    if pairs[i, 0].cpu().numpy() in predicted_pairs[pairs_mask, 1]:
                        summer += 1

                    pbar.update(1)

            self.siamese_accuracy = summer / pairs.size(0)
            logging.info(
                f'Siamese accuracy of maching for the tops of top{self.topk}: {self.siamese_accuracy * 100:.2f}%')
            logging.info(f'Combined mathing of top{self.topk} contains {np.sum(top_pairs_array>0)} elements')

        return top_pairs_array, predicted_pairs

    def run_bipartite_matching(self, pairs_array):
        logging.info('Running maximum bipartite matching.')
        pairs = self.iohandler.results['pairs']

        if self.siamese_accuracy > self.config.env.bipartite_threshold:
            predicted_pairs = linear_sum_assignment(pairs_array, maximize=True)[1]

            linear_sum_accuracy = np.sum(predicted_pairs == pairs[:, 0].cpu().numpy()) / pairs.size(0)

            self.accuracy = linear_sum_accuracy
            logging.info(f'linear_sum_acc: {linear_sum_accuracy * 100:.2f}%')
        else:
            self.accuracy = 0
            predicted_pairs = None

        return predicted_pairs