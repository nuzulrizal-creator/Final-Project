import os
import tensorflow as tf
from tensorflow import keras

def inspect_model():
    model_path_keras = os.path.join("models", "amazon_mlp_model.keras")
    model_path_h5 = os.path.join("models", "amazon_mlp_model.h5")
    
    if os.path.exists(model_path_keras):
        print(f"Loading model from {model_path_keras}...")
        model = keras.models.load_model(model_path_keras)
    elif os.path.exists(model_path_h5):
        print(f"Loading model from {model_path_h5}...")
        model = keras.models.load_model(model_path_h5)
    else:
        print("Model file not found!")
        return

    print("\n--- MODEL SUMMARY ---")
    model.summary()
    
    print("\n--- MODEL INPUTS ---")
    print("Input shape:", model.input_shape)
    for i, inp in enumerate(model.inputs):
        print(f"Input {i}: name={inp.name}, shape={inp.shape}, dtype={inp.dtype}")
        
    print("\n--- MODEL OUTPUTS ---")
    for i, out in enumerate(model.outputs):
        print(f"Output {i}: name={out.name}, shape={out.shape}, dtype={out.dtype}")
        
    print("\n--- LAYER NAMES & SHAPES ---")
    for layer in model.layers:
        print(f"Layer {layer.name}: type={layer.__class__.__name__}, input_shape={layer.input_shape}, output_shape={layer.output_shape}")

if __name__ == "__main__":
    inspect_model()
