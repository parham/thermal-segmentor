
""" 
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @professor  Professor Xavier Maldague
    @organization: Laval University

    @name           Unsupervised Image Segmentation by Backpropagation
    @journal        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    @year           2018
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation
    @citation       Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
    @info           the code is based on the implementation presented in the mentioned repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init
from skimage import segmentation

from phm import KWIterativeNNSegmentator


class Kanezaki2018Net (nn.Module):

    def __init__(self, net_config, num_dim):
        super(Kanezaki2018Net, self).__init__()

        # Set the model's config based on provided configuration
        self.config = net_config
        nChannel = self.config['num_channels']
        nConv = self.config['num_conv_layers']

        self.conv1 = nn.Conv2d(
            num_dim, nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append(nn.Conv2d(nChannel, nChannel,
                                        kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannel))
        self.conv3 = nn.Conv2d(
            nChannel, nChannel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        nConv = self.config['num_conv_layers']

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class Kanezaki2018Segmentator(KWIterativeNNSegmentator):
    def __init__(self,
                 seg_config,
                 model=None,
                 optimizer=None,
                 ) -> None:
        super().__init__(seg_config=seg_config, model=model, optimizer=optimizer)
        self.l_inds = None
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def pre_segment(self, img):
        # slic
        labels = segmentation.slic(
            img, compactness=self.compactness, n_segments=self.superpixel_regions)
        labels = labels.reshape(img.shape[0] * img.shape[1])
        u_labels = np.unique(labels)
        self.l_inds = []
        for i in range(len(u_labels)):
            self.l_inds.append(np.where(labels == u_labels[i])[0])

        # train
        if self.model is None:
            self.model = Kanezaki2018Net(self.model_config, img.shape[2])

        super().pre_segment(img)

    def calc_loss(self, img, output, target):
        # superpixel refinement
        im_target = target.data.cpu().numpy()
        for i in range(len(self.l_inds)):
            labels_per_sp = im_target[self.l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[self.l_inds[i]] = u_labels_per_sp[np.argmax(hist)]

        return self.loss_fn(output, target)

# class Kanezaki2018Segmentator(Segmentator):
#     def __init__(self, seg_config) -> None:
#         super().__init__(seg_config['segmentation'])
#         self.model_config = seg_config['model']

#     def segment(self, img):
#         nChannel = self.model_config['num_channels']

#         data = torch.from_numpy(np.array([img.transpose( (2, 0, 1) ).astype('float32')/255.]))
#         if self.use_cuda:
#             data = data.cuda()
#         data = Variable(data)

#         # slic
#         labels = segmentation.slic(img, compactness = self.compactness, n_segments=self.superpixel_regions)
#         labels = labels.reshape(img.shape[0] * img.shape[1])
#         u_labels = np.unique(labels)
#         l_inds = []
#         for i in range(len(u_labels)):
#             l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

#         # train
#         model = Kanezaki2018Net(self.model_config, data.size(1))
#         if self.use_cuda:
#             model.cuda()

#         model.train()
#         loss_fn = torch.nn.CrossEntropyLoss()
#         optimizer = optim.SGD(model.parameters(), lr = self.learning_rate, momentum=0.9)
#         label_colours = np.random.randint(255,size=(100,3))

#         seg_result = None
#         seg_num_classes = 0

#         for batch_idx in range(self.iteration):
#             # forwarding
#             optimizer.zero_grad()
#             output = model(data)[0]
#             output = output.permute(1, 2, 0).contiguous().view( -1, nChannel)
#             _, target = torch.max( output, 1 )
#             im_target = target.data.cpu().numpy()
#             nLabels = len(np.unique(im_target))

#             seg_result = im_target
#             seg_num_classes = nLabels

#             if self.visualize:
#                 im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
#                 im_target_rgb = im_target_rgb.reshape(img.shape).astype(np.uint8)
#                 cv2.imshow("output", im_target_rgb)
#                 cv2.waitKey(10)

#             # superpixel refinement
#             # TODO: use Torch Variable instead of numpy for faster calculation
#             for i in range(len(l_inds)):
#                 labels_per_sp = im_target[l_inds[i]]
#                 u_labels_per_sp = np.unique(labels_per_sp)
#                 hist = np.zeros(len(u_labels_per_sp))
#                 for j in range(len(hist)):
#                     hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
#                 im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]

#             target = torch.from_numpy(im_target)
#             if self.use_cuda:
#                 target = target.cuda()

#             target = Variable(target)
#             loss = loss_fn(output, target)
#             loss.backward()
#             optimizer.step()

#             #print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
#             print (batch_idx, '/', self.iteration, ':', nLabels, loss.item())

#             if nLabels <= self.min_classes:
#                 print ("nLabels", nLabels, "reached minLabels", self.min_classes, ".")
#                 break

#         return seg_result, seg_num_classes
