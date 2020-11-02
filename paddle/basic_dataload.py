import os
import cv2
import numpy as np
import paddle
import paddle.fluid as fluid

class BasicDataLoader():
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle
        self.data_list = self.read_list()

    def read_list(self):
        data_list = []

    def preprocess(self, data, label):
        h, w, c = data.shape
        g_gt, w_gt = label.shape
        assert h == g_gt, "Error"
        assert w == w_gt, "Error"
        if self.transform:
            data, label = self.transform(data, label)
        label = [:, :, np.newaxis]

    def __len__(self):
        return len(self.data_list)
    # 在调用 a = A(),中 使用 a()会调用__call__()方法
    def __call__(self, *args, **kwargs):
        for data_path, label_path in self.data_list:
            data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BRG2RGB)

def main():
    batch_size = 5
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        basic_dataloader = BasicDataLoader()

        dataloader = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        dataloader.set_sample_generator(basic_dataloader,
                                        batch_size=batch_size,
                                        places=place)
        num_epoch = 2
        for epoch in range(1, num_epoch+1):
            print(f"Epoch [{}]")
