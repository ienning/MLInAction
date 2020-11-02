import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Pool2D
import numpy as np
np.set_printoptions(precision=2)

class BasicModel(fluid.dygraph.Layer):
    def __init__(self, num_class=59):
        super(BasicModel, self).__init__()
        self.pool = Pool2D(pool_size=2, pool_stride=2)
        self.conv = Conv2D(num_channels=3, num_filter=num_class, filter_size=1)
    def forward(self, inputs):
        x = self.pool(inputs)
        x = fluid.layers.interpolate(x, out_shape=input.shape[2::])
        x = self.conv(x)
        return x

def main():
    place = paddle.fluid.CPUPlace()
    # palce = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.gurard(palce):
        model = BasicModel(num_classes=59)
        model.eval()
        input_data = np.random.rand(1, 3, 8, 8).astype(np.float32)
        print("Input data shape:", input_data.shape)
        input_data = to_variable(input_data)
        output_data = model(input_data)
        output_data = output_data.numpy()
        print("Output_data is ", output_data)