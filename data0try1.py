import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, InputLayer, BatchNormalization, Dropout, Conv1D, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm


# 特征转换为极坐标形式
def convert_to_polar(H):
    amplitude = np.abs(H)
    phase = np.angle(H)
    return np.concatenate([amplitude, phase], axis=-1)


# 增强数据集并存储到磁盘
def augment_data_to_file(X, Y, augmentation_factor, output_dir):
    """
    逐批增强数据并保存到磁盘。
    :param X: 原始特征数据 (numpy array)
    :param Y: 原始标签数据 (numpy array)
    :param augmentation_factor: 每个样本的增强次数
    :param output_dir: 增强后的数据存储路径
    """
    os.makedirs(output_dir, exist_ok=True)
    augmented_X_file = os.path.join(output_dir, "augmented_X.npy")
    augmented_Y_file = os.path.join(output_dir, "augmented_Y.npy")

    with open(augmented_X_file, 'wb') as fX, open(augmented_Y_file, 'wb') as fY:
        for i in tqdm(range(len(X)), desc="Augmenting Data"):
            for _ in range(augmentation_factor):
                noise = np.random.normal(0, 0.01, X[i].shape).astype(np.float32)
                augmented_X = X[i] + noise
                augmented_Y = Y[i]

                np.save(fX, augmented_X)
                np.save(fY, augmented_Y)

    print(f"增强数据已保存到目录: {output_dir}")
    return augmented_X_file, augmented_Y_file


# 从磁盘加载增强数据
def load_augmented_data_from_file(augmented_X_file, augmented_Y_file, batch_size):
    """
    逐批加载增强数据。
    :param augmented_X_file: 增强特征数据文件路径
    :param augmented_Y_file: 增强标签数据文件路径
    :param batch_size: 每次加载的批次大小
    :yield: 每次加载的 (X_batch, Y_batch)
    """
    with open(augmented_X_file, 'rb') as fX, open(augmented_Y_file, 'rb') as fY:
        while True:
            X_batch = []
            Y_batch = []
            try:
                for _ in range(batch_size):
                    X_batch.append(np.load(fX))
                    Y_batch.append(np.load(fY))
                yield np.array(X_batch), np.array(Y_batch)
            except EOFError:
                if X_batch and Y_batch:
                    yield np.array(X_batch), np.array(Y_batch)
                break


# 构建深度学习模型
def build_improved_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape,)))

    # 增加卷积层处理空间关系
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())

    # 增加更多的全连接层
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")

    # 文件路径配置
    cfg_path = r"D:\data0\Dataset0CfgData1.txt"
    anch_pos_path = r"D:\data0\Dataset0InputPos1.txt"
    output_dir = r"D:\data0"

    slice_num = 20
    batch_size = 5000  # 每次处理的样本数量
    scaler = StandardScaler()

    # 初始化空列表，用于存储逐批处理后的数据
    X_batches = []

    for slice_idx in tqdm(range(slice_num), desc="Processing Slices"):
        slice_file = f'{output_dir}/H_slice_{slice_idx}.npy'
        H_tmp = np.load(slice_file).astype(np.complex64)  # 加载后直接转换为 complex64

        # 转换为极坐标形式
        H_polar = convert_to_polar(H_tmp)

        # 展平 4D 数组为 2D 数组
        total_samples, port_num, sc_num, ant_num = H_polar.shape
        X_tmp = H_polar.reshape(total_samples, port_num * sc_num * ant_num).astype(np.float32)

        # 按批次标准化数据
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            X_batch_scaled = scaler.fit_transform(X_tmp[start:end])  # 标准化当前批次
            X_batches.append(X_batch_scaled)  # 存储已标准化的批次

    # 将所有批次数据合并为一个数组
    X = np.concatenate(X_batches, axis=0)

    # 确保最终结果保持为 float32
    X = X.astype(np.float32)

    # 准备标签（Y），使用锚点的真实位置
    ground_truth_path = r"D:\data0\Dataset0GroundTruth1.txt"
    ground_truth = np.loadtxt(ground_truth_path, delimiter=' ')
    Y = ground_truth[:, 1:]

    # 数据增强并存储到磁盘
    augment_output_dir = r"D:\augmented_data"
    augmented_X_file, augmented_Y_file = augment_data_to_file(X, Y, augmentation_factor=2, output_dir=augment_output_dir)

    # 从磁盘加载增强数据（分批处理）
    print("分批加载增强数据...")
    for X_batch, Y_batch in load_augmented_data_from_file(augmented_X_file, augmented_Y_file, batch_size=1000):
        print(f"Loaded batch with shape X: {X_batch.shape}, Y: {Y_batch.shape}")

    # 模型训练略（可根据实际需求对加载数据进行训练）
