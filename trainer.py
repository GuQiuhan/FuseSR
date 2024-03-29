import os
import math
from decimal import Decimal

import utility
import pdb
import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8


    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, sample in enumerate(self.loader_train):
            frame_i, frame_i_1, frame_i_2 = (
                sample['frame_i'],
                sample['frame_i_1'],
                sample['frame_i_2']
            )
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(frame_i, frame_i_1, frame_i_2)
            loss = self.loss(sr, frame_i.hr_gbuffer)
            loss.backward()
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        target = self.model.module if self.args.n_GPUs > 1 else self.model
        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
        )
        

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
