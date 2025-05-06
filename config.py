# config.py

# 数据相关
DATA_DIR    = './cifar-10-batches-py'
URL_PREFIX  = 'https://www.cs.toronto.edu/~kriz/'

# 训练相关
BATCH_SIZE  = 128     # 调小一点，提高稳定性
EPOCHS      = 60
LR          = 0.0005
MOMENTUM    = 0.9
WEIGHT_DECAY = 0.001  # ⭐⭐ 这个就是之前缺少的⭐⭐

# 其他
NUM_CLASSES = 10
CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
