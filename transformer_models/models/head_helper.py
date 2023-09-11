#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        train_act_func=None,
        test_act_func=None,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)
        
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # activation for training.
        if train_act_func == "softmax":
            self.train_act = nn.Softmax(dim=4)
        elif train_act_func == "sigmoid":
            self.train_act = nn.Sigmoid()
        elif train_act_func is None:
            self.train_act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as a training activation"
                "function.".format(train_act_func)
            )

        # activation for evaluation and testing.
        if test_act_func == "softmax":
            self.test_act = nn.Softmax(dim=4)
        elif test_act_func == "sigmoid":
            self.test_act = nn.Sigmoid()
        elif test_act_func is None:
            self.test_act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as a evaluation activation"
                "function.".format(test_act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.projection(x)

        if self.training:
            x = self.train_act(x)
        else:
            x = self.test_act(x)

        x = x.view(x.shape[0], -1)
        return x


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        train_act_func=None,
        test_act_func=None,
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()

        if isinstance(num_classes, list):
            assert len(num_classes) == 1
            num_classes = num_classes[0]
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # activation for training.
        if train_act_func == "softmax":
            self.train_act = nn.Softmax(dim=1)
        elif train_act_func == "sigmoid":
            self.train_act = nn.Sigmoid()
        elif train_act_func is None:
            self.train_act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as a training activation"
                "function.".format(train_act_func)
            )

        # activation for evaluation and testing.
        if test_act_func == "softmax":
            self.test_act = nn.Softmax(dim=1)
        elif test_act_func == "sigmoid":
            self.test_act = nn.Sigmoid()
        elif test_act_func is None:
            self.test_act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as a evaluation activation"
                "function.".format(test_act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)
        if self.training:
            x = self.train_act(x)
        else:
            x = self.test_act(x)
        return x
    



class MultiTaskSlowfastHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        train_act_func=None,
        test_act_func=None,
    ):
        """
        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        projs = []
        for n in num_classes:
            projs.append(nn.Linear(sum(dim_in), n, bias=True))
        self.projections = nn.ModuleList(projs)

        # activation for training.
        if train_act_func == "softmax":
            self.train_act = nn.Softmax(dim=4)
        elif train_act_func == "sigmoid":
            self.train_act = nn.Sigmoid()
        elif train_act_func is None:
            self.train_act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as a training activation"
                "function.".format(train_act_func)
            )

        # activation for evaluation and testing.
        if test_act_func == "softmax":
            self.test_act = nn.Softmax(dim=4)
        elif test_act_func == "sigmoid":
            self.test_act = nn.Sigmoid()
        elif test_act_func is None:
            self.test_act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as a evaluation activation"
                "function.".format(test_act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = []
        for projection in self.projections:
            x.append(projection(x))

        x = torch.cat(x, 1)  # (N, T, H, W, C) -> (N, num_heads, T, H, W, C)

        if self.training:
            x = self.train_act(x)
        else:
            x = self.test_act(x)

        x = x.view(x.shape[0], x.shape[1], -1)  # (N, num_heads, C)
        return x


class MultiTaskMViTHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        train_act_func=None,
        test_act_func=None,
    ):
        super(MultiTaskMViTHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        projs = []
        for n in num_classes:
            projs.append(nn.Linear(sum(dim_in), n, bias=True))
        self.projections = nn.ModuleList(projs)

        # activation for training.
        if train_act_func == "softmax":
            self.train_act = nn.Softmax(dim=4)
        elif train_act_func == "sigmoid":
            self.train_act = nn.Sigmoid()
        elif train_act_func is None:
            self.train_act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as a training activation"
                "function.".format(train_act_func)
            )

        # activation for evaluation and testing.
        if test_act_func == "softmax":
            self.test_act = nn.Softmax(dim=4)
        elif test_act_func == "sigmoid":
            self.test_act = nn.Sigmoid()
        elif test_act_func is None:
            self.test_act = nn.Identity()
        else:
            raise NotImplementedError(
                "{} is not supported as a evaluation activation"
                "function.".format(test_act_func)
            )

    def forward(self, inputs):
        # Perform dropout.

        feat = inputs
        if hasattr(self, "dropout"):
            feat = self.dropout(feat)

        x = []
        for projection in self.projections:
            # print(feat.shape, projection)
            x.append(self.act(projection(feat)))

        x = torch.cat(x, 1)  # (N, num_heads, C)
        return x
