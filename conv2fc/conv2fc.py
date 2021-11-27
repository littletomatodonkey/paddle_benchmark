import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import numpy as np


class SimFC(nn.Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None):
        super().__init__()
        self.conv = nn.Conv2D(
            in_features,
            out_features,
            1,
            weight_attr=weight_attr,
            bias_attr=bias_attr)
        self.out_features = out_features

    def forward(self, x):
        shape = x.shape
        x = x.reshape([np.prod(shape[:-1]), shape[-1], 1, 1])
        x = self.conv(x)
        x = x.reshape(shape[:-1] + [self.out_features])
        return x


class BaseFCLayerNorm(nn.Layer):
    def __init__(self, input_channel, outoput_channel):
        super().__init__()
        self.fc = nn.Linear(input_channel, outoput_channel)
        self.ln = nn.LayerNorm(outoput_channel)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        return x


class SimFCLayerNorm(nn.Layer):
    def __init__(self, input_channel, outoput_channel):
        super().__init__()
        self.fc = SimFC(input_channel, outoput_channel)
        self.ln = nn.LayerNorm(outoput_channel)

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        return x


if __name__ == "__main__":
    in_ch = 768
    out_ch = 512
    x = paddle.rand([32, in_ch])
    base_model = BaseFCLayerNorm(in_ch, out_ch)
    feat_model = SimFCLayerNorm(in_ch, out_ch)

    # run
    y1 = base_model(x)
    y2 = feat_model(x)
    print(y1.shape)
    print(y2.shape)

    # save
    base_model = paddle.jit.to_static(
        base_model,
        input_spec=[InputSpec(
            shape=[None, in_ch], dtype='float32')])
    paddle.jit.save(base_model, "base")

    feat_model = paddle.jit.to_static(
        feat_model,
        input_spec=[InputSpec(
            shape=[None, in_ch], dtype='float32')])
    paddle.jit.save(base_model, "feat")
