from layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, SoftmaxCrossEntropy, BatchNorm2D, Dropout
import numpy as np

class ImprovedCNN:
    def __init__(self):
        self.layers = [
            Conv2D(3, 32, kernel_size=3, padding=1),
            BatchNorm2D(32),
            ReLU(),
            Conv2D(32, 32, kernel_size=3, padding=1),
            BatchNorm2D(32),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(32, 64, kernel_size=3, padding=1),
            BatchNorm2D(64),
            ReLU(),
            Conv2D(64, 64, kernel_size=3, padding=1),  # 增加一层卷积
            BatchNorm2D(64),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(64*8*8, 256),
            ReLU(),
            Dropout(0.3),   # Dropout 减弱一点
            Dense(256, 10)
        ]
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x, training=True):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm2D)):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def step(self, lr, momentum):
        for layer in self.layers:
            if hasattr(layer, 'step'):
                layer.step(lr, momentum)

    def predict(self, x, batch_size=256):
        preds = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i+batch_size]
            out = self.forward(xb, training=False)
            preds.append(np.argmax(out, axis=1))
        return np.concatenate(preds, axis=0)
