# -*- coding:utf-8 -*-
from __future__ import print_function , division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets , models , transforms
import matplotlib.pyplot as plt
import time
import os
import pylab

#data process
data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224) ,
        transforms.RandomHorizontalFlip() ,
        transforms.ToTensor() ,
        transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
    ]) ,
    'val' : transforms.Compose([
        transforms.Resize(256) ,
        transforms.CenterCrop(224) ,
        transforms.ToTensor() ,
        transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229 , 0.224 , 0.225])
    ]) ,
}

data_dir = 'hymenoptera_data'
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir , x) , data_transforms[x]) for x in ['train' , 'val']}
dataloders = {x : torch.utils.data.DataLoader(image_datasets[x] , batch_size = 4 , shuffle = True , num_workers = 0) for x in ['train' , 'val']}
dataset_sizes = {x : len(image_datasets[x]) for x in ['train' , 'val']}
class_names = image_datasets['train'].classes
print(class_names)
use_gpu = torch.cuda.is_available()
#show several images
def imshow(inp , title = None):
    inp = inp.numpy().transpose((1 , 2 , 0))
    mean = np.array([0.485 , 0.456 , 0.406])
    std = np.array([0.229 , 0.224 , 0.225])
    inp = std * inp + mean
    inp = np.clip(inp , 0 , 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    pylab.show()
    plt.pause(0.001)

inputs , classes = next(iter(dataloders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out , title = [class_names[x] for x in classes])
#train the model
def train_model(model , criterion , optimizer , scheduler , num_epochs = 25):

    since = time.time()
    best_model_wts = model.state_dict()  #Returns a dictionary containing a whole state of the module.
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch , num_epochs - 1))
        print('-' * 10)
        #set the mode of model
        for phase in ['train' , 'val']:
            if phase == 'train':
                scheduler.step()  #about lr and gamma
                model.train(True)  #set model to training mode
            else:
                model.train(False)  #set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            #Iterate over data
            for data in dataloders[phase]:
                inputs , labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    lables = Variable(labels)
                optimizer.zero_grad()
                #forward
                outputs = model(inputs)
                _ , preds = torch.max(outputs , 1)
                loss = criterion(outputs , labels)
                #backward
                if phase == 'train':
                    loss.backward()  #backward of gradient
                    optimizer.step()  #strategy to drop
                running_loss += loss.item()
                running_corrects += torch.sum(preds.data == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase , epoch_loss , epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

#visualizing the model predictions
def visualize_model(model , num_images = 6):
    images_so_far = 0
    fig = plt.figure()

    for i , data in enumerate(dataloders['val']):
        inputs , labels = data
        if use_gpu:
            inputs , labels = Variable(inputs.cuda()) , Variable(labels.cuda())
        else:
            inputs , labels = Variable(inputs) , Variable(labels)

        outputs = model(inputs)
        _ , preds = torch.max(outputs.data , 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2 , 2 , images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

#Finetuning the convnet
from torchvision.models.resnet import model_urls
model_urls['resnet18'] = model_urls['resnet18'].replace('https://' , 'http://')
model_ft = models.resnet18(pretrained = True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs , 2)
if use_gpu:
    model_ft = model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters() , lr = 0.001 , momentum = 0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft , step_size = 7 , gamma = 0.1)
#start finetuning
model_ft = train_model(model_ft , criterion , optimizer_ft , exp_lr_scheduler , num_epochs = 25)
torch.save(model_ft.state_dict() , 'C:/Users/chLi/Desktop/pytorch/resnet18.pth')
visualize_model(model_ft)