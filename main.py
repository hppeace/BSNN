import os
import random
import warnings

import ray
from ray import tune, train

import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import StepLR

from net.snns import SNN
from net.optim import SWATS
from dataset import SetLoader
from net.utils import set_seed


# 运行一次这个函数执行一个epoch的train(因为ray库里面有train所以不能取名trian)
def fit(model, loader, optim, crit, nclasses):
    model.cuda()
    model.train()

    running_data = running_correct = running_loss = 0
    for spikes, labels in loader:
        num_data = len(labels)
        spikes, labels = spikes.float().cuda(), torch.nn.functional.one_hot(labels,nclasses).float().cuda()
        model.zero_grad()
        optim.zero_grad()
        prediction = model(spikes)
        loss = crit(prediction.float(), labels.float())
        loss.backward()

        optim.step()

        num_correct = prediction.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()
        running_correct += num_correct
        running_loss += loss.item()
        running_data += num_data

    return running_correct / running_data, running_loss / running_data


# 运行一次这个函数执行一个epoch的test
def test(model, loader, crit, nclasses):
    model.cuda()
    model.eval()

    with torch.no_grad():
        running_data = running_correct = running_loss = 0
        for spikes, labels in loader:
            num_data = len(labels)
            spikes, labels = spikes.float().cuda(), torch.nn.functional.one_hot(labels,nclasses).float().cuda()
            model.zero_grad()

            prediction = model(spikes)
            loss = crit(prediction.float(), labels.float())

            num_correct = prediction.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()
            running_correct += num_correct
            running_loss += loss.item()
            running_data += num_data

    return running_correct / running_data, running_loss / running_data


#  运行一次这个函数执行对应config的完整训练与测试(epochs)
def trainable(config):
    warnings.filterwarnings("ignore")
    # 将固定参数放在了trainable内，因为ray库要求trainable好像只能传一个参数
    set_seed(config["seed"])

    # 初始化
    trainloader, testloader = SetLoader(config)
    crit = nn.MSELoss()
    network = SNN(config)
    optim = SWATS(network.parameters(), lr=config["lr"])
    scheduler = StepLR(optim, step_size=config["step_size"], gamma=config["lr_gamma"])

    best_accuracy = 0.0
    best_epoch = 0

    start = 0
    epochs = config["epochs"]

    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start = checkpoint_dict["epoch"]
            best_accuracy = checkpoint_dict["best_accuracy"]
            best_epoch = checkpoint_dict["best_epoch"]
            network.load_state_dict(checkpoint_dict["model_state"])

    for epoch in range(start, epochs):
        training_accuracy, training_loss = fit(network, trainloader, optim, crit, config["nclasses"])
        testing_accuracy, testing_loss = test(network, testloader, crit, config["nclasses"])
        scheduler.step()

        if testing_accuracy > best_accuracy:
            best_accuracy = testing_accuracy
            best_epoch = epoch

        metrics = {
            "train_accuracy": training_accuracy,
            "testing_accuracy": testing_accuracy,
            "training_loss": training_loss,
            "testing_loss": testing_loss,
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
        }

        train.report(metrics=metrics)


def main():
    storage_path = "/home/peace/code/Log"
    exp_name = "BSNN_CIFAR10DVS_module_structures"

    config = {
        "seed":0,
        "T": 10,
        "epochs": 200,

        # optimizer  
        "optimizer": "SWATS",
        "lr": 5e-4,
        "step_size":100,
        "lr_gamma": 0.1,

        # neuron
        "threshold": 1.0,
        "decay": 0.5,
        "gamma": 0.8,

        # method
        # "binarize_weight": "ste_clip",
        # "norm_method": "tdBN",
        # "binarize_norm": False,
        # "pool_method": "max",
        # "net_structure":  "CXXPBN",
        "binarize_weight": "ste_clip",
        "norm_method": "tdBN",
        "binarize_norm": False,
        "pool_method": tune.grid_search(["max", "avg"]),
        "net_structure": tune.grid_search(["CBNPBN", "CBNPXX", "CXXPBN"]),

        # net
        "nclasses":10,
        "linearing_size": 10 * 10 * 128,

        #dataset
        "dataset": "CIFAR10DVS",
        "batch_size": 50,
    }

    # N-MNIST
    # config = {
    #     "seed":0,
    #     "T": 18,
    #     "epochs": 100,

    #     "optimizer": "SWATS",
    #     "lr": 1e-3,
    #     "step_size":50,
    #     "lr_gamma": 0.1,

    #     "binarize_weight": "ste_clip",
    #     "norm_method": "tdBN",
    #     "binarize_norm": False,
    #     "pool_method": "max",
    #     "net_structure":  "CXXPBN",

    #     "nclasses":10,
    #     "linearing_size": 6272,

    #     "dataset": "N-MNIST",
    #     "batch_size": 100,

    # }
    
    # DVSGesture
    # config = {
    #     "seed":0,
    #     "T": 25,
    #     "epochs": 100,

    #     "optimizer": "SWATS",
    #     "lr": 1e-3,
    #     "step_size":50,
    #     "lr_gamma": 0.1,

    #     "binarize_weight": "ste_clip",
    #     "norm_method": "tdBN",
    #     "binarize_norm": False,
    #     "pool_method": "max",
    #     "net_structure":  "CXXPBN",

    #     "nclasses":11,
    #     "linearing_size": 6272,

    #     "dataset": "DVSGesture",
    #     "batch_size": 24,

    # }

    # 为每个trainable函数分配运行资源
    trainable_with_resources = tune.with_resources(
        trainable=trainable,
        resources=train.ScalingConfig(
            trainer_resources={"CPU": 16, "GPU": 1},
            use_gpu=True,
            placement_strategy="SPREAD",
        ),
    )

    # 主进程初始化，管理所有trainable函数的运行
    path = os.path.join(storage_path, exp_name)
    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(
            path,
            param_space=config,
            trainable=trainable_with_resources,
            resume_unfinished=False,
        )
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=config,
            tune_config=tune.TuneConfig(
                # 网格搜索的每种组合执行一次
                num_samples=1,
            ),
            run_config=train.RunConfig(
                # Log文件存放的地方
                local_dir=storage_path,
                name=exp_name,
            ),
        )
    # 运行
    tuner.fit()

if __name__ == "__main__":
    # 1445789
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    # 限制总可用资源
    ray.init(
        num_cpus=70,
        num_gpus=4,
    )
    main()