from keras.models import load_model
from keras.losses import MeanSquaredError
import joblib
import numpy as np

# y�lb:�Pb
def convert_to_polar(H):
    amplitude = np.abs(H)
    phase = np.angle(H)
    return np.concatenate([amplitude, phase], axis=-1)

# �}v� Dataset1 �pn
def preprocess_dataset1(output_dir, slice_num, scaler):
    X_batches = []
    for slice_idx in range(slice_num):
        slice_file = f'{output_dir}/H_slice_{slice_idx}.npy'
        H_tmp = np.load(slice_file).astype(np.complex64)
        H_polar = convert_to_polar(H_tmp)
        total_samples, port_num, sc_num, ant_num = H_polar.shape
        X_tmp = H_polar.reshape(total_samples, port_num * sc_num * ant_num).astype(np.float32)
        X_batch_scaled = scaler.transform(X_tmp)
        X_batches.append(X_batch_scaled)
    X = np.concatenate(X_batches, axis=0)
    return X

if __name__ == "__main__":
    print("<<< (��}�!�� Dataset3 �L�K >>>")

    # !��
    model_path = r"D:\data0\Dataset0InputData1\final_model.h5"

    # Dataset3 �Mn
    dataset1_dir = r"D:\slice3\slice3-finish"
    slice_num = 20
    scaler_path = r"D:\slice3\slice3-finish\scaler.pkl"

    # �}�h
    scaler = joblib.load(scaler_path)
    print("�h���}")

    # �}!�K��_1�p
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
    print("!����}vͰ�")

    # � Dataset3 �pn
    X_dataset1 = preprocess_dataset1(dataset1_dir, slice_num, scaler)
    print(f"Dataset3 pn��b�:: {X_dataset1.shape}")

    # (!��L�K
    predictions = model.predict(X_dataset1)
    print(f"�K�Ӝb�:: {predictions.shape}")

    # �X�KӜ0��
    output_file = r"D:\slice3\Dataset3_Predicted_Positions.txt"
    np.savetxt(output_file, predictions, fmt='%.4f %.4f')
    print(f"�KӜ��X0: {output_file}")
