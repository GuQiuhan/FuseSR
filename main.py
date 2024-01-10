import torch

import utility
import model
import loss
from options import args
from trainer import Trainer
import dataloader

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)     # setting the log and the train information

# 未使用checkpoint
if checkpoint.ok:
    custom_dataset = dataloader.CustomDataset(data_dir=args.data_train, start_frame=args.start_frame, end_frame=args.end_frame)
    loader = torch.utils.data.DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=True)
    my_model = model.H_Net.HNet(args.scale)
    my_loss = loss.CustomLoss(lambda_p=0.5, lambda_s=0.05)
    t = Trainer(args, loader, my_model, my_loss, checkpoint)
    while not t.terminate():

        t.train()
        #t.test()

    checkpoint.done()
