# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:37:20 2020

@author: rishi
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batchSize = 64
imgSize = 64

transform = transforms.Compose([transforms.Resize(imgSize),
                                transforms.ToTensor(),
                                transforms.Normalize(tuple([0.5]*3),
                                               tuple([0.5]*3))])
dataset = dset.CIFAR10(root ='./data',
                       download = True,
                       transform = transform)
# Load data
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size = batchSize,
                                         shuffle=True,
                                         num_workers = 0)

# Weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining generator
class G(nn.Module):
# The neural Net for generator
  def __init__(self): #Constructor
    super(G, self).__init__()
    self.main = nn.Sequential(
        nn.ConvTranspose2d(100, 512, 4, 1, 0,bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
        nn.Tanh()
        )
    
  def forward(self, input):
    output = self.main(input)
    return output
# Network for Generator
netG = G()
netG.apply(weights_init)


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

netD = D()
netD.apply(weights_init)

# Training the model
loss = nn.BCELoss()
optmizerD = optim.Adam(netD.parameters(),
                      0.0002,betas=(0.5, 0.999))
optmizerG = optim.Adam(netG.parameters(),
                      0.0002,betas=(0.5, 0.999))

netG.cuda()
netD.cuda()

for epoch in range(50):
    for i,data in enumerate(dataloader,0):

        # Update the weights of Discriminator
        netD.zero_grad()

        real, _ = data
        input = Variable(real).cuda()

        target = Variable(torch.ones(input.size()[0])).cuda()

        output = netD(input)
        output = output.cuda()
        err = loss(output, target).cuda()

        noise = Variable(torch.randn((input.size()[0], 100, 1, 1))).cuda()
        fake = netG(noise)
        fake = fake.cuda()
        target = Variable(torch.zeros(input.size()[0])).cuda()
        output = netD(fake.detach())
        output = output.cuda()
        err_fake = loss(output, target).cuda()

        errTotal = err + err_fake
        errTotal.backward()
        optmizerD.step()

        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = netD(fake)
        output = output.cuda()
        errG = loss(output, target)

        errG.backward()
        optmizerG.step()

        print("[{0}/{1}][{2}/{3}] Loss_D: {4: .4f}, Loss_G: {5: .4f}".format(epoch, 50,
                                                                 i, len(dataloader),
                                                                 errTotal.item(),
                                                                 errG.item()))
        if i % 100 == 0:
            vutils.save_image(real,"{}/real_samples.jpeg".format("./results"),
                              normalize=True)
            fake = netG(noise)
            fake = fake.cuda()
            vutils.save_image(fake,"{0}/fake_samples_epoch_{1: 03d}.jpeg".format("./results",epoch),
                              normalize=True)