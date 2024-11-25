import os
import numpy as np
from matplotlib import pyplot as plt
from Dataset0InputData1.data0try1 import build_improved_model, load_augmented_data_from_file


# ��!����p/	��e
def build_improved_model(input_shape):
    from keras.models import Sequential
    from keras.layers import Dense, InputLayer, BatchNormalization, Dropout, Conv1D, Flatten, Reshape
    from keras.regularizers import l2

    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape,)))

    # ���elb:	�
    model.add(Reshape((input_shape // 128, 128)))  # G��*7,	 128 *y�

    # w�B
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())

    # ����hޥB
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


# �!�
model = build_improved_model(input_shape=104448)

# y�}�:pnv��!�
batch_size = 1000  # �!�} 1000 *7,
epochs_per_batch = 1  # �y!�� * epochtS���
augmented_X_file = r"D:\augmented_data\augmented_X.npy"
augmented_Y_file = r"D:\augmented_data\augmented_Y.npy"

# B���6��t*�:pnƄ!p
for epoch in range(10):  # ;q���� 10 !
    print(f"Epoch {epoch + 1}")
    # ���y�}pnv��!�
    for X_batch, Y_batch in load_augmented_data_from_file(augmented_X_file, augmented_Y_file, batch_size):
        # ��!�
        history = model.fit(X_batch, Y_batch, epochs=epochs_per_batch, batch_size=32, verbose=1)

# �����
plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss vs Validation Loss')
plt.show()

# �X!�
model.save('final_model.h5')  # �X!�0SM�U
print("!���X0 'final_model.h5'")
