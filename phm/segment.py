
""" 
    @name       Base classes for Segmentation
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from skimage import segmentation

import time

from phm import Configurable, running_time

class Segmentator(Configurable):

    def __init__(self, seg_config) -> None:
        super().__init__(seg_config)

    def __call__(self, img) -> dict:
        return self.segment(img)

    def segment_(self, img) -> dict:
        """ For compatibility purposes, it is better to implement the segmentators to handle OpenCV images. """
        pass

    @running_time
    def segment(self, img) -> dict:
        return self.segment_(img)


class KWIterativeNNSegmentator(Segmentator):
    def __init__(self, 
        seg_config,
        model = None,
        optimizer = None,
    ) -> None:
        super().__init__(seg_config['segmentation'])
        self.model_config = seg_config['model']
        self.nChannel = self.model_config['num_channels']
        self.model = model 
        self.optimizer = optimizer
        self.init_()

    def init_ (self):
        pass

    def calc_loss(self, img, output, target):
        pass

    def pre_segment(self, img):
        lr = self.learning_rate if hasattr(self, 'learning_rate') else 0.1
        momentum = self.momentum if hasattr(self, 'momentum') else 0.9
        if self.optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.label_colours = np.random.randint(255,size=(100,3))

    def segment_(self, img) -> dict:
        # Convert image to numpy data
        img_data = np.array([img.transpose((2, 0, 1)).astype('float32')/255.])
        data = torch.from_numpy(img_data)
        if self.use_cuda:
            data = data.cuda()
        data = Variable(data)

        self.pre_segment(img)

        # Create an instance of the model and set it to learn
        if self.use_cuda:
            self.model.cuda()
        self.model.train()

        result = {}
        seg_result = None
        seg_num_classes = 0
        seg_step_time = 0
        img_w = img.shape[0]
        img_h = img.shape[1]
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)  
        for batch_idx in range(self.iteration):
            t = time.time()
            self.optimizer.zero_grad()
            output = self.model(data)[0,:,0:img_w,0:img_h]

            output_orig = output.permute(1, 2, 0).contiguous()
            output = output_orig.view(-1, self.nChannel)
            # output = output.permute(1, 2, 0).contiguous().view( -1, self.nChannel)
            _, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            seg_result = im_target
            seg_num_classes = nLabels

            if self.visualize:
                im_target_rgb = np.array([[c, c, c] for c in im_target]) # [c, c, c] # self.label_colours[ c % 100 ]
                im_target_rgb = im_target_rgb.reshape(img.shape).astype(np.uint8)
                cv2.imshow("output", im_target_rgb)
                cv2.waitKey(10)

            target = torch.from_numpy(im_target)
            if self.use_cuda:
                target = target.cuda()

            target = Variable(target)
            loss = self.calc_loss(img, output, target)

            loss.backward()
            self.optimizer.step()

            print (batch_idx, '/', self.iteration, ':', nLabels, loss.item())

            seg_step_time += time.time() - t
            if nLabels <= self.min_classes:
                print ("nLabels", nLabels, "reached minLabels", self.min_classes, ".")
                break
        
        result['iteration_time'] = seg_step_time
        result['segmentation'] = seg_result
        result['num_classes'] = seg_num_classes
        return result