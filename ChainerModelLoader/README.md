### サンプル内にて読み込んでいるファイルの作成コード
```Python
import chainer
from chainer import serializers
import chainer.functions as F
import chainer.links as L
import numpy as np

class NN(chainer.Chain):
    def __init__(self):
        initial_W1 = np.array([[[[1.0, 0.5,0.0],[0.5, 0.0,-0.5],[0.0, -0.5,-1.0]]],[[[0.0, -0.1, 0.1],[-0.3, 0.4, 0.7],[0.5, -0.2, 0.2]]]], dtype=np.float32)
        initial_b1 = np.array([0.5,1.0], dtype=np.float32)
        initial_W2 = np.array([[[[-0.1, 0.6],[0.3, -0.9]],[[0.7, 0.9],[-0.2, -0.3]]],[[[-0.6, -0.1],[0.3, 0.3]],[[-0.5, 0.8],[0.9, 0.1]]]], dtype=np.float32)
        initial_b2 = np.array([0.1, 0.9], dtype=np.float32)
        initial_W3 = np.array([[0.5, 0.3, 0.4, 0.2, 0.6, 0.1, 0.4, 0.3],[0.6,0.4,0.9,0.1,0.5,0.2,0.3,0.4]], dtype=np.float32)
        initial_b3 = np.array([0.01, 0.02], dtype=np.float32)
        initial_W4 = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32)
        initial_b4 = np.array([0.02, 0.01], dtype=np.float32)
        super(NN, self).__init__(
            conv1=L.Convolution2D(1,2,3,initialW=initial_W1,initial_bias=initial_b1),
            conv2=L.Convolution2D(2,2,2,initialW=initial_W2,initial_bias=initial_b2),
            fl3=L.Linear(8,2,initialW=initial_W3,initial_bias=initial_b3),
            b1=L.BatchNormalization(2),
            fl4=L.Linear(2,2,initialW=initial_W4,initial_bias=initial_b4)
        )

    def __call__(self, x):
        h_conv1 = F.relu(self.conv1(x))
        h_pool1 = F.max_pooling_2d(h_conv1, 2)
        h_conv2 = F.relu(self.conv2(h_pool1))
        h_pool2 = F.max_pooling_2d(h_conv2, 2)
        h_fc1 = F.relu(self.fl3(h_pool2))
        y = self.fl4(h_fc1)
        return y

model = NN()
serializers.save_npz("ChainerModel.npz", model)
```

