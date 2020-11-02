import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear
from vgg import VGG16BN


class FC():
    def __init__(self, num_classes=59):
        self.input = input

    def forward(self, inputs):
        x = self.layer1(inputs)

        return x
def main():
    print("ss")

if __name__ == '__main__':
    main()