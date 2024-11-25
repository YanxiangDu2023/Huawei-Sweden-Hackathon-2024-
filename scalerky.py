import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# y�lb:�Pb
def convert_to_polar(H):
    amplitude = np.abs(H)
    phase = np.angle(H)
    return np.concatenate([amplitude, phase], axis=-1)


if __name__ == "__main__":
    print("<<<  � Dataset3 pnv�X�h >>>")

    # ���Mn
    output_dir = r"D:\slice3\slice3-finish"
    slice_num = 20  # Dataset0 � slice ��p�
    batch_size = 5000  # �!�7,p�

    # ��h
    scaler = StandardScaler()

    # �zh(�X�y�pn
    X_batches = []

    for slice_idx in tqdm(range(slice_num), desc="Processing Slices"):
        # �} slice ��
        slice_file = f'{output_dir}/H_slice_{slice_idx}.npy'
        H_tmp = np.load(slice_file).astype(np.complex64)  # �}ppn
        H_polar = convert_to_polar(H_tmp)  # lb:�Pb

        # Us 4D p�: 2D p�
        total_samples, port_num, sc_num, ant_num = H_polar.shape
        X_tmp = H_polar.reshape(total_samples, port_num * sc_num * ant_num).astype(np.float32)

        # 	y!�pn
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            X_batch_scaled = scaler.fit_transform(X_tmp[start:end])  # �SMy!
            X_batches.append(X_batch_scaled)  # X����y!

    # @	y!pnv: *�ty��5
    X = np.concatenate(X_batches, axis=0)

    # �X�h
    scaler_path = r"D:\slice3\slice3-finish\scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"�h���X0: {scaler_path}")
