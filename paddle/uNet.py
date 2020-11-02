import os
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid

from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Layer
class UNet(Layer):
    self.down1 = Encoder(num_channels = 3, num_filter=64)
    self.down2 = Encoder(num_channels = 64, num_filter=64)
    self.down3 = Encoder(num_channels = 128, num_filter=64)
    self.down4 = Encoder(num_channels = 256, num_filter=64)

    self.mid_conv1 = Conv2D(512, 1024, filter_size=1, paddle=0, stride=1)
    self.mid_bn1 = BatchNorm(1024, act="relu")
    self.mid_conv2 = Conv2D(1024, 1024, filter_size=1, paddle=0, stride=1)
    self.mid_bn2 = BatchNorm(1024, act="relu")

    self.down1 = Encoder(num_channels=3, num_filter=64)


class Encoder():
    def __init__(self, num_channels, num_filters):
        self.conv1 = Conv2D(num_channels,
                            num_filters,
                            filter_size=3,
                            stride=1,
                            padding=1)
        self.bn1 = BatchNorm(num_filter, act="relu")

        self.conv1 = Conv2D(num_channels,
                        num_filters,
                        filter_size=3,
                        stride=1,
                        padding=1)
        self.bn1 = BatchNorm(num_filter, act="relu")
        self.pool = Pool2D(pool_size=2, pool_stride=2, pool_type="max", ceil_mode=True)

    def forward(self,inputs):
        x = self.conv1

class Decoder():
    def __init__(self):
    def forward(self, inputs_prev, inputs):


def main():
    print("2")

if __name__ == '__main__':
    main()

