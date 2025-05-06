import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape).astype(np.float32)
            return x * self.mask / (1 - self.dropout_rate)
        else:
            return x

    def backward(self, dout):
        return dout * self.mask / (1 - self.dropout_rate)

class BatchNorm2D:
    def __init__(self, channels, eps=1e-5, momentum=0.9):
        self.gamma = np.ones(channels)
        self.beta = np.zeros(channels)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)
        self.mean = None
        self.var = None

    def forward(self, x, training=True):
        if training:
            self.mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            self.var = np.var(x, axis=(0, 2, 3), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var.squeeze()
        else:
            self.mean = self.running_mean.reshape(1, -1, 1, 1)
            self.var = self.running_var.reshape(1, -1, 1, 1)
        
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma.reshape(1, -1, 1, 1) * self.x_norm + self.beta.reshape(1, -1, 1, 1)

    def backward(self, dout):
        N, C, H, W = dout.shape
        dgamma = np.sum(dout * self.x_norm, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        self.dgamma = dgamma.squeeze()
        self.dbeta = dbeta.squeeze()

        dx_norm = dout * self.gamma.reshape(1, -1, 1, 1)
        dvar = np.sum(
            dx_norm * (self.x_norm * -0.5) * np.power(self.var + self.eps, -1.5),
            axis=(0, 2, 3), keepdims=True
        )
        dmean = np.sum(
            dx_norm * (-1 / np.sqrt(self.var + self.eps)),
            axis=(0, 2, 3), keepdims=True
        ) + dvar * np.mean(-2 * (self.x_norm), axis=(0, 2, 3), keepdims=True)

        dx = dx_norm / np.sqrt(self.var + self.eps)
        dx += (2 / (N * H * W)) * dvar * self.x_norm
        dx += (1 / (N * H * W)) * dmean

        return dx

class Conv2D:
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.pad = padding
        self.W = np.random.randn(out_ch, in_ch, kernel_size, kernel_size) * np.sqrt(2/(in_ch * kernel_size * kernel_size))
        self.b = np.zeros(out_ch)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        N, C, H, W = x.shape
        K = self.W.shape[2]
        out_h = (H + 2*self.pad - K) // self.stride + 1
        out_w = (W + 2*self.pad - K) // self.stride + 1
        x_pad = np.pad(x, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        col = self.im2col(x_pad, K, self.stride)
        col_W = self.W.reshape(self.W.shape[0], -1) 
        out = col.dot(col_W.T) + self.b  
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) 
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        N, F, out_h, out_w = dout.shape
        K = self.W.shape[2]
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, F) 
        self.db = np.sum(dout_reshaped, axis=0)
        self.dW = dout_reshaped.T.dot(self.col).reshape(self.W.shape)
        dcol = dout_reshaped.dot(self.col_W)
        dx = self.col2im(dcol, self.x.shape, K, self.stride, self.pad)
        return dx

    def step(self, lr, momentum):
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb

    @staticmethod
    def im2col(x, K, stride):
        N, C, H, W = x.shape
        out_h = (H - K) // stride + 1
        out_w = (W - K) // stride + 1

        col = np.zeros((N, C, K, K, out_h, out_w),dtype=x.dtype)

        for y in range(K):
            y_max = y + stride*out_h
            for x_ in range(K):
                x_max = x_ + stride*out_w
                col[:, :, y, x_, :, :] = x[:, :, y:y_max:stride, x_:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col

    @staticmethod
    def col2im(col, x_shape, K, stride, pad):
        N, C, H, W = x_shape
        H_pad, W_pad = H + 2*pad, W + 2*pad
        x_pad = np.zeros((N, C, H_pad, W_pad))
        out_h = (H_pad - K) // stride + 1
        out_w = (W_pad - K) // stride + 1

        col = col.reshape(N, out_h, out_w, C, K, K).transpose(0, 3, 4, 5, 1, 2)

        for y in range(K):
            y_max = y + stride*out_h
            for x_ in range(K):
                x_max = x_ + stride*out_w
                x_pad[:, :, y:y_max:stride, x_:x_max:stride] += col[:, :, y, x_, :, :]

        if pad == 0:
            return x_pad
        return x_pad[:, :, pad:-pad, pad:-pad]


class MaxPool2D:
    def __init__(self, kernel, stride):
        self.kernel, self.stride = kernel, stride

    def forward(self, x):
        self.x = x
        B, C, H, W = x.shape
        PH = PW = self.kernel
        S = self.stride
        # 计算输出尺寸
        out_h = (H - PH) // S + 1
        out_w = (W - PW) // S + 1
        x_reshaped = x.reshape(B, C, out_h, S, out_w, S)
        # 对重塑后的数据沿着步长维度取最大值
        out = x_reshaped.max(axis=3).max(axis=4)
        # 通过广播将输出重复到原始大小，然后比较找出最大值位置
        self.max_mask = (x == np.repeat(
            np.repeat(out, S, axis=2), S, axis=3))
        
        return out

    def backward(self, dout):
        # 创建一个与输入相同形状的零矩阵
        dx = np.zeros_like(self.x)
        
        # 将梯度重复扩展到与输入相同的尺寸
        dout_repeated = np.repeat(np.repeat(dout, self.stride, axis=2), 
                                 self.stride, axis=3)
        
        dx[self.max_mask] = dout_repeated[self.max_mask]
        
        return dx

class Flatten:
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, dout):
        return dout.reshape(self.orig_shape)

class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2/in_dim)
        self.b = np.zeros((1,out_dim))
        self.vW = np.zeros_like(self.W); self.vb = np.zeros_like(self.b)
    def forward(self, x):
        self.x = x
        return x.dot(self.W) + self.b
    def backward(self, dout):
        self.dW = self.x.T.dot(dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout.dot(self.W.T)
    def step(self, lr, momentum):
        self.vW = momentum*self.vW - lr*self.dW
        self.vb = momentum*self.vb - lr*self.db
        self.W  += self.vW
        self.b  += self.vb

class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def backward(self, dout):
        return dout * self.mask

class SoftmaxCrossEntropy:
    def forward(self, x, y_true):
        # x: (N, C), y_true: one-hot (N, C)
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = ex / np.sum(ex, axis=1, keepdims=True)
        self.y = y_true
        loss = -np.sum(y_true * np.log(self.probs + 1e-9)) / x.shape[0]
        return loss

    def backward(self):
        N = self.y.shape[0]
        return (self.probs - self.y) / N
