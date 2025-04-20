# -*- coding: utf-8 -*-
"""Practice_3.ipynb

# Practice exercises after Lecture 3
This notebook contains the practice exercises with instructions and explanations.

You will explore different aspects related to data: quality, collection, and loading.

Work through the cells below in sequential order, executing each cell as you progress. Throughout the notebook, you will encounter instructions marked with the words **YOUR CODE HERE** followed by **raise NotImplementedError()**. You will have to substitute  *raise NotImplementedError()* with your own code.
Follow the instructions and write the code to complete the tasks.

Along the way, you will also find questions. Try to reflect on the questions before/after running the code.

This notebook was developed at the [Idiap Research Institute](https://www.idiap.ch) by [Alina Elena Baia](mailto:alina.baia.idiap.ch>), [Darya Baranouskaya](mailto:darya.baranouskaya.idiap.ch) and [Olena Hrynenko](mailto:olena.hrynenko.idiap.ch) (equal contribution). Some pieces of code were adopted from the notebooks developed at the  [Idiap Research Institute](https://www.idiap.ch) by [Olivier Canévet](mailto:olivier.canevet@idiap.ch).

Any reproduction or distribution of this document, in whole or in part, is prohibited unless permission is granted by the authors.

<!--
SPDX-FileCopyrightText: Copyright (c) 2019-2024 Idiap Research Institute <contact@idiap.ch>
SPDX-FileContributor: Alina Elena Baia <alina.baia.idiap.ch>
SPDX-FileContributor: Darya Baranouskaya <darya.baranouskaya.idiap.ch>
SPDX-FileContributor: Olena Hrynenko <olena.hrynenko.idiap.ch>
SPDX-FileContributor: Olivier Canévet <olivier.canevet@idiap.ch>
-->
"""

import getpass
import os
import random
import re
import tarfile
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils
import torchvision.transforms as transforms
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torchvision.models as models
#import funtions from another file in the same directory
from checkpoint_utils import save_checkpoint, load_checkpoint
import argparse


def parse_args():
    #when working with python files from console it's better to specify
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    return args.dataset_path, args.results_path


DATADIR, RESULTS_DIR = parse_args()
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
#directory to store model checkpoints to
if not os.path.exists(Path(RESULTS_DIR) / 'checkpoints'):
    os.mkdir(Path(RESULTS_DIR) / 'checkpoints')

'''
Summary of changes: 
1) since now we are running code in python and on the server instead of printing results and images
to the output like in .ipynb we are going to store images and save model checkpoints to correspoding folder RESULTS_PATH. 
Similarly the dataset will be loaded from the folder DATADIR.
To correctly specify this folders they are goign to be passed as arguments when running a commandline commands
 'python3 practice_3_simplified.py --dataset_path <path_to_the_dataset> --results_path <path_to_the_results>', 
where you need to replace <path_to_the_dataset> and <path_to_the_results> with corresponding paths on the computer where you are running the command.

2) Storing checkpoints on the model, optimiser, epoch and loss. 
Storing checkpoints is extermely important when you are running jobs on server. If you don't store the results of running the model,
 you are going to lose them after training. Additionally, it is useful to store intermediate checkpoints during training 
 and not only the final model since in case if the training gets interrupted you should be able to restore the training from the intermediate checkpoint and 
 won't have to start the training from scratch.
'''


"""## Loading the data

##### 3.1 Recalling how to load the data from Practice 2.

We are going to load a dataset composed of images of the following fruits: apple, banana, bean, carrot, grape, onion, pear, potato, strawberry, tomato.

The dataset has the following structure:

```text
fruit_dataset/
|-- train/
|   |-- apple/
|   |   |-- 000.jpg
|   |   |-- 001.jpg
|   |   |-- ...
|   |-- banana/
|   |   |-- 000.jpg
|   |   |-- 001.jpg
|   |   |-- ...
|
|-- val/
|   |   |-- 040.jpg
|   |   |-- 041.jpg
|   |   |-- ...
|   |-- banana/
|   |   |-- 040.jpg
|   |   |-- 041.jpg
|   |   |-- ...
```

which can be loaded as is with the ImageFolder class of PyTorch.
Check out the documentation for more details: https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
"""

# This piece of code is adopted from the notebooks developed at the Idiap Research Institute by Olivier Canévet.
# For efficient usage of the hardware resources when running on JupyterHub (EPFL),
# we will limit the number of threads.

if re.search(r'^https://.*noto.*\.epfl\.ch$', os.environ.get("EXTERNAL_URL", "")) != None:
    num_threads_limit = 2
else:
    num_threads_limit = torch.get_num_threads()
print(f"Limiting the number of threads to {num_threads_limit}")
torch.set_num_threads(num_threads_limit)
print(f"PyTorch is using {torch.get_num_threads()} threads")

_ = torch.set_flush_denormal(True)


# Checking that your dataset directory exists

if not os.path.exists(Path(DATADIR) / "fruit_dataset" / "train"):
    raise Exception(f'Dataset not found. Please download the dataset first. '
                    f'It should be stored in the {Path(DATADIR) / "fruits_dataset"} directory')

"""You should copy the dataset folder fruits_dataset to the DATADIR path:
- the training images are in the folder ```<DATADIR>/ fruit_dataset / train``
- the validation images are in the folder ```<DATADIR>/ fruit_dataset / valid```
"""

#path to the train and val folders
train_dir = Path(DATADIR) / "fruit_dataset" / "train"
valid_dir = Path(DATADIR) / "fruit_dataset" / "val"


#name of the classes
classes = ("apple", "banana", "bean", "carrot", "grape", "onion", "pear",
           "potato", "strawberry", "tomato")
NUM_CLASSES = len(classes) #apple, banana, bean, carrot, grape, onion, pear, potato, strawberry, tomato.

# batch size
batch_size = 10

"""In Pytorch, when you load a dataset, you can pass a set of transformations to perform on the images before feeding them to a network. Usually, the transformations consist of: resizing and cropping the image, cast to a tensor, normalizing the image, etc.

Here you can find the available transformations in Pytorch: https://pytorch.org/vision/stable/transforms.html
"""

# Adaptation of the code from Practice 2
# mean and std for normalisation are chosen based on ImageNet statistics

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_set = datasets.ImageFolder(train_dir, transform)
valid_set = datasets.ImageFolder(valid_dir, transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

# Visualization of images

'''
Get some random training images.
A quick explanation of iter() and next():
https://stackoverflow.com/questions/62549990/what-does-next-and-iter-do-in-pytorchs-dataloader
'''

def plot_images_from_dataset(images, labels):
    batch_size = images.shape[0]
    fig = plt.figure(figsize=(8, 10))
    for idx in np.arange(int(batch_size)):
        ax = fig.add_subplot(5, int(batch_size/5), idx+1, xticks=[], yticks=[])
        img = np.squeeze(images[idx])
        img = img.to(torch.float64)
        # revert normalisation
        img = img * np.expand_dims(np.array([0.229, 0.224, 0.225]), (1, 2)) + np.expand_dims(np.array([0.485, 0.456, 0.406]), (1, 2))
        img = np.clip(img, 0, 1)
        plt.imshow(np.transpose(img, (1,2,0)))
        ax.set_title("{}({})".format(classes[labels[idx]],str(labels[idx].item())),
                    color=("blue"))

dataiter = iter(train_loader) #iter() returns an iterator
images, labels = next(dataiter) # call next() on the iterator to get the first iteration

plot_images_from_dataset(images, labels)
plt.savefig(Path(RESULTS_DIR) / "fruit_dataset.png")
plt.close()

# class distribution

print("How many images in the train set: ", len(train_set.targets))


# the training set is perfectly balanced: same number of images for all classes

"""## Metrics

##### 3.2 Introducing simple metrics, such as accuracy and F1-score.
"""

def f1(preds, target):
    return f1_score(target, preds, average='macro')

def acc(preds, target):
    return accuracy_score(target, preds)


"""## Model

##### 3.3 Loading pretrained network

You will fine-tune AlexNet: have a look at these 2 papers:

- [Original paper reaching a 15% top-5 error rate](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [How to parallelize the training across multiple GPUs](https://arxiv.org/abs/1404.5997)

The architecture of AlexNet is very similar to the LeNet5 convolutional network, except that it has more parameters and takes as input images of size `224x224`.

<div style="max-width:80%; margin:auto; padding:10px;">



</div>

Load AlexNet from the pytorch `models` module. By default, the model will be downloaded in `${TORCH_HOME}`. If not set, the default location is `~/.cache/torch`. You can find some information on how to load an existing architecture [here](https://pytorch.org/vision/stable/models.html), in particular, pay attention to the `weights` argument, and not using the deprecated `pretrained` one.
"""

class FruitNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        '''
        Load AlexNet
        '''
        self.model = models.alexnet(weights=weights)


        '''
        Change the layers in classifier part
        '''

        self.model.classifier[1] = nn.Linear(9216, 128)
        self.model.classifier[4] = nn.Linear(128, 128)
        self.model.classifier[6] = nn.Linear(128, 10)


    def freeze_feature_layers(self):
        '''
        Freeze the feature extraction part. Use a
        for loop over "self.features" parameters.
        '''

        for param in self.model.features:
            param.requires_grad_(False)


    def forward(self, x):
        return self.model(x)

"""Let's check that the wrapper does what it is needed:"""

# Load a non trained model
assert_model = FruitNet()
assert assert_model.model.classifier[1].weight.size() == torch.Size([128, 9216]), "Should be 128 x 9216"
assert assert_model.model.classifier[4].weight.size() == torch.Size([128, 128]), "Should be 128 x 128"
assert assert_model.model.classifier[6].weight.size() == torch.Size([10, 128]), "Should be 10 x 128"
assert assert_model.model.features[0].weight.requires_grad, "When creating the model, the layers should not be frozen"
assert_model.freeze_feature_layers()
assert not assert_model.model.features[0].weight.requires_grad, "After calling `freeze_feature_layers`, the layers should be frozen"
assert assert_model.model.classifier[6].weight.requires_grad, "The classifier should not be frozen as we will train it"
assert_model1 = FruitNet(weights=None)
assert_model2 = FruitNet(weights="DEFAULT")
assert not torch.equal(assert_model1.model.features[0].weight, assert_model2.model.features[0].weight), "Did you use the input `weights` argument, and not hardcoding weights when loading alexnet? Or did you use the `pretrained` flag?"


"""## Train cycle

##### 3.5 Implementing the training and evaluation functions, logging and plotting the metrics.

We are now prepared to write the function for our training. We will firstly write a function that will train our model for one epoch. We say that we train the model for one epoch, when we have passed all the data in the training dataset exactly once.
"""

'''
- set model into training state
- reset the gradients,
- compute forward pass,
- compute loss,
- compute gradients,
- update the parameters,
- compute model predictions,
- compute and log the metrics,
'''

def train_epoch(model, optimizer, criterion, metrics, train_loader, device):
    '''
    device = torch.device('cuda') or torch.device('cpu') if no GPU available
    '''
    # we activate the training state for the model
    model.train()

    # initialize epoch loss and metrics
    epoch_loss = 0
    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    # we iterate over the batches
    for batch_num, (x_batch, y_batch) in tqdm(enumerate(train_loader)):
        data = x_batch.to(device)
        target = y_batch.to(device)

        '''
        Clean previously computed gradients
        '''
        optimizer.zero_grad()

        '''
        Forward pass
        '''
        outputs = model(data)

        '''
        Compute loss
        loss = ...
        '''
        loss = criterion(outputs, target)

        '''
        Compute gradients and optimize the model parameters
        '''
        loss.backward()
        optimizer.step()

        '''
        Compute predictions
        Store predictions in variable preds
        No gradients should be propagated at this step
        '''
        with torch.no_grad():
            _, preds = torch.max(outputs, 1)

        '''
        Compute metrics
        Note: no gradients should be propagated at this step
        '''
        with torch.no_grad():
            for k in epoch_metrics.keys():
                epoch_metrics[k] += metrics[k](preds.cpu(), target.cpu())

        # log loss statistics
        epoch_loss += loss.item()

    # for the epoch loss, we take the average of the losses computed over the mini-batches
    epoch_loss /= len(train_loader)

    # for the epoch loss, we compute the average of the metrics over the mini-batches
    for k in epoch_metrics.keys():
          epoch_metrics[k] /= len(train_loader)

    print('train Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))

    return epoch_loss,  epoch_metrics

"""After we trained the model for one epoch, we need to record its performance on unseen data."""


def evaluate(model, criterion, metrics, test_loader, device):

    # we activate the evaluation state for the model
    model.eval()

    # initialize epoch loss and metrics
    epoch_loss = 0
    epoch_metrics = dict(zip(metrics.keys(), torch.zeros(len(metrics))))

    for batch_num, (x_batch, y_batch) in enumerate(test_loader):
        with torch.no_grad(): # since we do not update the model, we do not compute gradients
            data = x_batch.to(device)
            target = y_batch.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)
            _, preds = torch.max(outputs, 1)

            for k in epoch_metrics.keys():
                epoch_metrics[k] += metrics[k](preds.cpu(), target.cpu())

            epoch_loss += loss.item()

    epoch_loss /= len(test_loader)

    for k in epoch_metrics.keys():
          epoch_metrics[k] /= len(test_loader)

    print('eval Loss: {:.4f}, '.format(epoch_loss),
          ', '.join(['{}: {:.4f}'.format(k, epoch_metrics[k]) for k in epoch_metrics.keys()]))
    return epoch_loss,  epoch_metrics

"""Below are some utils functions that are used to plot the loss curves and update the loss log."""

def plot_training(train_loss, test_loss, metrics_names, train_metrics_logs, test_metrics_logs):
    fig, ax = plt.subplots(1, len(metrics_names) + 1, figsize=((len(metrics_names) + 1) * 5, 5))

    ax[0].plot(train_loss, c='blue', label='train')
    ax[0].plot(test_loss, c='orange', label='test')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

    for i in range(len(metrics_names)):
        ax[i + 1].plot(train_metrics_logs[i], c='blue', label='train')
        ax[i + 1].plot(test_metrics_logs[i], c='orange', label='test')
        ax[i + 1].set_title(metrics_names[i])
        ax[i + 1].set_xlabel('epoch')
        ax[i + 1].legend()

    plt.savefig(Path(RESULTS_DIR) / "training loss and metrics.jpg")

def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
    '''
    - metrics_names: the keys/names of the logged metrics
    - metrics_log: existing metrics log that will be updated
    - new_metrics_dict: epoch_metrics output from train_epoch and evaluate functions
    '''
    for i in range(len(metrics_names)):
        curr_metric_name = metrics_names[i]
        metrics_log[i].append(new_metrics_dict[curr_metric_name])
    return metrics_log

"""Training for one epoch is unlikely to be helpful, which is why we usually train the model for a number of epochs.

**Note**: you have to use validation for each step of training, but now we will focus only on the toy example and will track the performance on test set.
"""

def train_cycle(model, optimizer, criterion, metrics, train_loader, test_loader, n_epochs, device,
                store_checkpoint_for_every_epoch, start_epoch=0):
    train_loss_log,  test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    test_metrics_log = [[] for i in range(len(metrics))]


    for epoch in range(start_epoch, n_epochs):
        print("Epoch {0} of {1}".format(epoch, n_epochs - 1))
        train_loss, train_metrics = train_epoch(model, optimizer, criterion, metrics, train_loader, device)

        test_loss, test_metrics = evaluate(model, criterion, metrics, test_loader, device)

        train_loss_log.append(train_loss)
        train_metrics_log = update_metrics_log(metrics_names, train_metrics_log, train_metrics)

        test_loss_log.append(test_loss)
        test_metrics_log = update_metrics_log(metrics_names, test_metrics_log, test_metrics)

        plot_training(train_loss_log, test_loss_log, metrics_names, train_metrics_log, test_metrics_log)

        save_checkpoint(model, optimizer, epoch, loss=train_loss, checkpoint_path = Path(RESULTS_DIR) / "checkpoints/checkpoint.pth",
                        store_checkpoint_for_every_epoch=store_checkpoint_for_every_epoch)
    return train_metrics_log, test_metrics_log

"""##### 3.6 Training a simple network

It is now time to fine-tune the network. The first few layers will be frozen and will not be updated in the backward pass. These layers correspond to image features which we will see later in the course. They have been trained on 1 million images of ImageNet, so the features are meaningful for image classification and don't need to be updated.

The last three linear layers that you have created will be optimized. You will fine-tune the network with: the cross-entropy loss, and the stochastic gradient descent.
"""

torch.manual_seed(42)

model = FruitNet(weights="IMAGENET1K_V1")
model.freeze_feature_layers()

learning_rate = 0.001

metrics = {'ACC': acc, 'F1-weighted': f1}


# initialize the stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#if the checkpoint already exists this function will load checkpoint for the model and optimiser so that the training can be continued
# if no checkpoint is found the training is going to start from scratch
saved_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path = Path(RESULTS_DIR) / "checkpoints/checkpoint.pth")
if saved_epoch == 0:
    start_epoch = 0
else:
    start_epoch = saved_epoch + 1  #if the checkpoint from the epoch saved_epoch is stored, we want to start the training from the next epoch

criterion = nn.CrossEntropyLoss()

N_EPOCHS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model.to(device)
criterion.to(device)


train_metrics_log, test_metrics_log = train_cycle(model, optimizer, criterion, metrics, train_loader, valid_loader,
                                                  n_epochs=N_EPOCHS, device=device, store_checkpoint_for_every_epoch=False,
                                                  start_epoch=start_epoch)

