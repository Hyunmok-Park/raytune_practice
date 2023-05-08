from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import time

from net import Net
from data_loader import make_data, get_loader_segment


def train(config):
    current_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    result_save_path = os.path.join('result', current_time)
    os.makedirs(result_save_path, exist_ok=True)

    model = Net(config['l1'], config['l2'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.MSELoss(reduction='mean')

    model.to(device)

    train_loader = get_loader_segment('/Users/user/Desktop/raytune_practice/dataset/train', 'train', config['batch_size'])
    val_loader = get_loader_segment('/Users/user/Desktop/raytune_practice/dataset/val', 'val', config['batch_size'])

    logs = []

    for epoch in range(config['epoch']):
        optimizer.zero_grad()
        train_loss = []
        val_loss = []

        model.train()
        for x, y in train_loader:
            x = x.to(device).type(torch.float32)
            y = y.to(device).type(torch.float32)
            pred = model(x)
            loss = criterion(pred, y)

            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_loss)
        if epoch % 10 == 0:
            print(f"EPOCH:{epoch} / loss: {train_loss}")
            logs.append(f"EPOCH:{epoch} / loss: {train_loss}")

        model.eval()
        for x, y in val_loader:
            x = x.to(device).type(torch.float32)
            y = y.to(device).type(torch.float32)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        path = os.path.join(result_save_path, 'checkpoint')
        torch.save(model.state_dict(), path)

        tune.report(loss=val_loss)

        with open(os.path.join(result_save_path, 'log.txt')) as f:
            for _ in logs:
                f.write(_)
        f.close()

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "epoch": tune.choice([50, 100])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)


if __name__ == '__main__':
    make_data('dataset', 500, 50, 20)
    # config = {
    #     'l1': 64,
    #     'l2': 64,
    #     'lr': 0.001,
    #     'batch_size': 16,
    #     'epoch': 100
    # }
    # train(config)
    main(10, 100, 1)

