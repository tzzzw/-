import numpy as np
from tqdm import tqdm
from config import BATCH_SIZE, EPOCHS, LR as INIT_LR, MOMENTUM, NUM_CLASSES, CLASS_NAMES, WEIGHT_DECAY  # 导入 WEIGHT_DECAY
from data_loader import load_data
from model1 import ImprovedCNN
from utils import one_hot, print_classification_report
import os
 
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

def train():
    x_train, y_train, x_test, y_test = load_data()
    x_train = np.ascontiguousarray(x_train)
    x_test = np.ascontiguousarray(x_test)
    y_train_oh = one_hot(y_train, NUM_CLASSES)

    model = ImprovedCNN()
    num_batches = x_train.shape[0] // BATCH_SIZE

    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    m_t = {}
    v_t = {}

    # 初始化 m_t 和 v_t
    for layer in model.layers:
        if hasattr(layer, 'W'):
            m_t[(layer, 'W')] = np.zeros_like(layer.W)
            v_t[(layer, 'W')] = np.zeros_like(layer.W)
        if hasattr(layer, 'b'):
            m_t[(layer, 'b')] = np.zeros_like(layer.b)
            v_t[(layer, 'b')] = np.zeros_like(layer.b)

    for epoch in range(1, EPOCHS + 1):
        # 余弦退火学习率
        lr = INIT_LR * 0.5 * (1 + np.cos(np.pi * (epoch - 1) / EPOCHS))  # 修改学习率策略

        idx = np.random.permutation(x_train.shape[0])
        x_train, y_train_oh = x_train[idx], y_train_oh[idx]
        epoch_loss = 0.0
        correct_train = 0

        with tqdm(total=num_batches, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch") as pbar:
            for i in range(num_batches):
                xb = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                yb = y_train_oh[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

                preds = model.forward(xb, training=True)
                loss = model.loss_fn.forward(preds, yb)
                epoch_loss += loss
                correct_train += np.sum(np.argmax(preds, axis=1) == np.argmax(yb, axis=1))

                dout = model.loss_fn.backward()
                model.backward(dout)

                step = epoch * num_batches + i + 1

                for layer in model.layers:
                    if hasattr(layer, 'W'):
                        # Adam 更新权重 W
                        m_t[(layer, 'W')] = beta1 * m_t[(layer, 'W')] + (1 - beta1) * layer.dW
                        v_t[(layer, 'W')] = beta2 * v_t[(layer, 'W')] + (1 - beta2) * (layer.dW ** 2)
                        m_hat = m_t[(layer, 'W')] / (1 - beta1 ** step)
                        v_hat = v_t[(layer, 'W')] / (1 - beta2 ** step)
                        # 梯度更新
                        layer.W -= lr * (m_hat / (np.sqrt(v_hat) + epsilon))
                        # 显式添加权重衰减 (L2正则化)
                        layer.W -= lr * WEIGHT_DECAY * layer.W  # 关键修改点

                    if hasattr(layer, 'b'):
                        # Adam 更新偏置 b（不添加权重衰减）
                        m_t[(layer, 'b')] = beta1 * m_t[(layer, 'b')] + (1 - beta1) * layer.db
                        v_t[(layer, 'b')] = beta2 * v_t[(layer, 'b')] + (1 - beta2) * (layer.db ** 2)
                        m_hat_b = m_t[(layer, 'b')] / (1 - beta1 ** step)
                        v_hat_b = v_t[(layer, 'b')] / (1 - beta2 ** step)
                        layer.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

                pbar.set_postfix({"loss": f"{loss:.4f}"})
                pbar.update(1)

        avg_loss = epoch_loss / num_batches
        train_acc = correct_train / (num_batches * BATCH_SIZE) * 100
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}%")

        if epoch % 2 == 0:
            y_pred = model.predict(x_test, batch_size=256)
            test_acc = np.mean(y_pred == y_test.flatten()) * 100
            print(f"Epoch {epoch} - Test Acc: {test_acc:.2f}%")

    y_pred = model.predict(x_test, batch_size=256)
    print_classification_report(y_test, y_pred, CLASS_NAMES)

if __name__ == '__main__':
    train()