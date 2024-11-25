from keras.models import load_model
from keras.losses import MeanSquaredError
import joblib
import numpy as np

# yÅlb:ÅPb
def convert_to_polar(H):
    amplitude = np.abs(H)
    phase = np.angle(H)
    return np.concatenate([amplitude, phase], axis=-1)

# †}vÑ Dataset1 Ñpn
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
    print("<<< (≠√}Ñ!ã˘ Dataset3 €LÑK >>>")

    # !ãÔÑ
    model_path = r"D:\data0\Dataset0InputData1\final_model.h5"

    # Dataset3 ÑMn
    dataset1_dir = r"D:\slice3\slice3-finish"
    slice_num = 20
    scaler_path = r"D:\slice3\slice3-finish\scaler.pkl"

    # †}∆h
    scaler = joblib.load(scaler_path)
    print("∆hÚü†}")

    # †}!ãK®ö_1˝p
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
    print("!ãÚü†}vÕ∞—")

    # Ñ Dataset3 Ñpn
    X_dataset1 = preprocess_dataset1(dataset1_dir, slice_num, scaler)
    print(f"Dataset3 pnÚÑåb∂:: {X_dataset1.shape}")

    # (!ã€LÑK
    predictions = model.predict(X_dataset1)
    print(f"ÑKå”úb∂:: {predictions.shape}")

    # ›XÑK”ú0áˆ
    output_file = r"D:\slice3\Dataset3_Predicted_Positions.txt"
    np.savetxt(output_file, predictions, fmt='%.4f %.4f')
    print(f"ÑK”úÚ›X0: {output_file}")
