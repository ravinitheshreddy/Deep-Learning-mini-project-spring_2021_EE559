import torch
from torch import nn
from torch.nn import functional as F


class fullyconnected(nn.Module):
    """
    Class for a fully connected network
    ...

    Attributes
    ----------
    use_weight_sharing_ : bool
        A boolean indicating if the network uses weight sharing
    use_auxiliary_loss_ : bool
        A boolean indicating if the network uses auxiliary losses for training

    Methods
    ----------
    forward(x)
        Performs the forward computation of the network.   
    """

    # Init method
    def __init__(self, use_weight_sharing_: bool, use_auxiliary_loss_: bool):
        """
        Initialises the network and accepts two arguments

        Parameters
        ----------
        use_weight_sharing_: bool
            Model uses weight sharing if set to True.
        use_auxiliary_loss_: bool
            Model uses auxiliary loss for training if set to True.
        """

        super().__init__()
        self.use_weight_sharing = use_weight_sharing_
        self.use_auxiliary_loss = use_auxiliary_loss_

        # For the first image X1
        self.x1_fc1 = nn.Linear(196, 50)
        self.x1_fc2 = nn.Linear(50, 10)

        # For the second image X2
        if not self.use_weight_sharing:
            # create the layers only if the network is not using weight sharing
            self.x2_fc1 = nn.Linear(196, 50)
            self.x2_fc2 = nn.Linear(50, 10)

        # For the combined feature vector of two images
        self.comp_fc1 = nn.Linear(20, 20)
        self.comp_fc2 = nn.Linear(20, 2)

        # Dropout
        self.dropout = nn.Dropout()

    # Forward method
    def forward(self, x: torch.Tensor):
        """
        Performs the forward computation of the network.

        Parameters
        ----------
        x: torch.Tensor
            A two channel tensor with size 14 x 14.
        """

        # Forward computation for first image X1
        x1 = x[:, 0:1]
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.dropout(F.relu(self.x1_fc1(x1)))
        x1 = F.relu(self.x1_fc2(x1))

        # Forward computation for second image X2
        x2 = x[:, 1:2]
        x2 = x2.reshape(x2.size(0), -1)

        if not self.use_weight_sharing:
            # use different layers if not using weight sharing
            x2 = self.dropout(F.relu(self.x2_fc1(x2)))
            x2 = F.relu(self.x2_fc2(x2))

        else:
            # use the same layers if using weight sharing
            x2 = self.dropout(F.relu(self.x1_fc1(x2)))
            x2 = F.relu(self.x1_fc2(x2))

        # Combining the feature vectors
        x = torch.cat((x1, x2), 1)
        x = F.relu(self.comp_fc1(x))
        x = self.comp_fc2(x)

        if self.use_auxiliary_loss:
            # return the output and the feature vector of the images if network uses auxiliary loss for training
            return x, x1, x2
        else:
            return x


class simpleCNN(nn.Module):
    """
    Class for a simple convolutional network
    ...

    Attributes
    ----------
    use_weight_sharing_ : bool
        A boolean indicating if the network uses weight sharing
    use_auxiliary_loss_ : bool
        A boolean indicating if the network uses auxiliary losses for training

    Methods
    ----------
    forward(x)
        Performs the forward computation of the network.   
    """

    def __init__(self, use_weight_sharing_: bool, use_auxiliary_loss_: bool):
        """
        Initialises the network and accepts two arguments

        Parameters
        ----------
        use_weight_sharing_: bool
            Model uses weight sharing if set to True.
        use_auxiliary_loss_: bool
            Model uses auxiliary loss for training if set to True.
        """

        super().__init__()
        self.use_weight_sharing = use_weight_sharing_
        self.use_auxiliary_loss = use_auxiliary_loss_
        self.conv_out = 10 * 6 * 6

        # For the first image X1
        self.x1_conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.x1_conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.x1_fc1 = nn.Linear(self.conv_out, 10)

        # For the second image X2
        if not self.use_weight_sharing:
            self.x2_conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
            self.x2_conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
            self.x2_fc1 = nn.Linear(self.conv_out, 10)

        # common
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # For the combined feature vector of two images
        self.comp_fc1 = nn.Linear(20, 20)
        self.comp_fc2 = nn.Linear(20, 2)

        # Dropout
        self.dropout = nn.Dropout()

    # Forward method
    def forward(self, x: torch.Tensor):
        """
        Performs the forward computation of the network.

        Parameters
        ----------
        x: torch.Tensor
            A two channel tensor with size 14 x 14.
        """

        # Forward computation for first image X1
        x1 = x[:, 0:1]
        x1 = self.dropout(F.relu(self.max_pool1(self.x1_conv1(x1))))
        x1 = self.dropout(F.relu(self.max_pool2(self.x1_conv2(x1))))
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.x1_fc1(x1)

        # Forward computation for second image X2
        x2 = x[:, 1:2]
        if not self.use_weight_sharing:
            # use different layers if not using weight sharing
            x2 = self.dropout(F.relu(self.max_pool1(self.x2_conv1(x2))))
            x2 = self.dropout(F.relu(self.max_pool2(self.x2_conv2(x2))))
            x2 = x2.reshape(x2.size(0), -1)
            x2 = self.x2_fc1(x2)
        else:
            # use same layers if using weight sharing
            x2 = self.dropout(F.relu(self.max_pool1(self.x1_conv1(x2))))
            x2 = self.dropout(F.relu(self.max_pool2(self.x1_conv2(x2))))
            x2 = x2.reshape(x2.size(0), -1)
            x2 = self.x1_fc1(x2)

        # Combining the feature vectors
        x = torch.cat((x1, x2), 1)
        x = F.relu(self.comp_fc1(x))
        x = self.comp_fc2(x)

        if self.use_auxiliary_loss:
            # return the output and the feature vector of the images if network uses auxiliary loss for training
            return x, x1, x2
        else:
            return x


# To be used in the Residual Network
class ResBlock(nn.Module):
    """
    Class for a residual block
    ...

    Attributes
    ----------
    nb_channels : int
        An integer indicating the number of channels in the input image
    kernel_size : int
        Size of the convolving kernel

    Methods
    ----------
    forward(x)
        Performs the forward computation of the network.   
    """

    def __init__(self, nb_channels: int, kernel_size: int):

        """
        Initialises the block and accepts two arguments

        Parameters
        ----------
        nb_channels : int
            An integer indicating the number of channels in the input image
        kernel_size : int
            Size of the convolving kernel
        """

        super().__init__()

        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        """
        Performs the forward computation of the block.

        Parameters
        ----------
        x: torch.Tensor
            A two channel tensor with size 14 x 14.
        """

        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y


class skipCNN(nn.Module):
    """
    Class for a Residual convolutional network
    ...

    Attributes
    ----------
    use_weight_sharing_ : bool
        A boolean indicating if the network uses weight sharing
    use_auxiliary_loss_ : bool
        A boolean indicating if the network uses auxiliary losses for training

    Methods
    ----------
    forward(x)
        Performs the forward computation of the network.   
    """

    # Init method
    def __init__(self, use_weight_sharing_: bool, use_auxiliary_loss_: bool):
        """
        Initialises the network and accepts two arguments

        Parameters
        ----------
        use_weight_sharing_: bool
            Model uses weight sharing if set to True.
        use_auxiliary_loss_: bool
            Model uses auxiliary loss for training if set to True.
        """

        super().__init__()
        self.use_weight_sharing = use_weight_sharing_
        self.use_auxiliary_loss = use_auxiliary_loss_
        self.rb_kernel_size = 3  # Odd size expected
        self.nb_blocks = 6

        # For the first image X1
        self.x1_conv0 = nn.Conv2d(1, 14, kernel_size=3)
        self.x1_resblocks = nn.Sequential(*(ResBlock(14, self.rb_kernel_size) for _ in range(self.nb_blocks)))
        self.x1_avg = nn.AvgPool2d(kernel_size=12)
        self.x1_fc = nn.Linear(14, 10)

        # For the second image X1
        if not self.use_weight_sharing:
            # create the layers only if the network is not using weight sharing
            self.x2_conv0 = nn.Conv2d(1, 14, kernel_size=3)
            self.x2_resblocks = nn.Sequential(*(ResBlock(14, self.rb_kernel_size) for _ in range(self.nb_blocks)))
            self.x2_avg = nn.AvgPool2d(kernel_size=12)
            self.x2_fc = nn.Linear(14, 10)

        # For the combined feature vector of two images
        self.comp_fc1 = nn.Linear(20, 20)
        self.comp_fc3 = nn.Linear(20, 2)

        # Dropout
        self.dropout = nn.Dropout()

    # Forward method
    def forward(self, x: torch.Tensor):
        """
        Performs the forward computation of the network.

        Parameters
        ----------
        x: torch.Tensor
            A two channel tensor with size 14 x 14.
        """

        # Forward computation for first image X1
        x1 = F.relu(self.x1_conv0(x[:, 0:1]))
        x1 = self.x1_resblocks(x1)
        x1 = F.relu(self.x1_avg(x1))
        x1 = x1.view(x1.size(0), -1)
        x1 = self.x1_fc(x1)

        # Forward computation for second image X2
        if not self.use_weight_sharing:
            # use different layers if not using weight sharing
            x2 = F.relu(self.x2_conv0(x[:, 1:2]))
            x2 = self.x2_resblocks(x2)
            x2 = F.relu(self.x2_avg(x2))
            x2 = x2.view(x2.size(0), -1)
            x2 = self.x2_fc(x2)
        else:
            # use same layers if using weight sharing
            x2 = F.relu(self.x1_conv0(x[:, 1:2]))
            x2 = self.x1_resblocks(x2)
            x2 = F.relu(self.x1_avg(x2))
            x2 = x2.view(x2.size(0), -1)
            x2 = self.x1_fc(x2)

        # Combining the feature vectors
        x = F.relu(self.comp_fc1(torch.cat((x1, x2), 1)))
        x = self.comp_fc3(x)

        if self.use_auxiliary_loss:
            # return the output and the feature vector of the images if network uses auxiliary loss for training
            return x, x1, x2
        else:
            return x
