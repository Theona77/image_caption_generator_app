import os
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json, Model
import json
import numpy as np # Used for dummy weights array
import h5py # Used for low-level H5 operations

# --- Configuration ---
MODEL_DIR = 'outputs/models'
ORIGINAL_CAPTION_MODEL_NAME = 'model.keras'
ORIGINAL_FEATURE_EXTRACTOR_NAME = 'feature_extractor.keras'
RESAVED_CAPTION_MODEL_NAME = 'model_resaved.keras'
RESAVED_FEATURE_EXTRACTOR_NAME = 'feature_extractor_resaved.keras'

# --- Paths ---
original_caption_path = os.path.join(MODEL_DIR, ORIGINAL_CAPTION_MODEL_NAME)
original_feature_path = os.path.join(MODEL_DIR, ORIGINAL_FEATURE_EXTRACTOR_NAME)
resaved_caption_path = os.path.join(MODEL_DIR, RESAVED_CAPTION_MODEL_NAME)
resaved_feature_path = os.path.join(MODEL_DIR, RESAVED_FEATURE_EXTRACTOR_NAME)

# --- FIX: Internal Class Mapping for custom_objects ---
class Policy:
    """Mock Policy class to satisfy DTypePolicy deserialization."""
    def __init__(self, name):
        self.name = name
    def get_config(self):
        return {'name': self.name}
    @classmethod
    def from_config(cls, config):
        return cls(config['name'])

# Use a lambda function for InputLayer mapping, as InputLayer cannot be mocked easily
# It prevents crashes when Model.from_config attempts to resolve 'InputLayer'
def InputLayerWrapper(*args, **kwargs):
    return tf.keras.layers.InputLayer(**kwargs)

# Custom objects dictionary for all serialization fixes
CUSTOM_OBJECTS = {
    # Crucial fix for DTypePolicy references
    "DTypePolicy": Policy, 
    "Policy": Policy, 
    
    # Failsafe class name/module replacements
    "Functional": Model,  
    "InputLayer": InputLayerWrapper,
    "keras.src.models.functional": Model,
    "keras.src.models.functional.Functional": Model, 
    "keras.layers.Functional": Model,
}

# --- Core Resave Function (Optimized for H5 extraction) ---

def resave_model_safely(original_path, resaved_path):
    print(f"\n--- Processing {os.path.basename(original_path)} ---")
    config_str = None
    weights = None

    if not os.path.exists(original_path):
        print(f"ERROR: Original model not found at {original_path}")
        return

    # 1. Extract Config String and Weights using low-level H5 read
    try:
        with h5py.File(original_path, 'r') as f:
            # The config is typically stored as JSON string under 'config'
            if 'config' in f.attrs:
                config_str = f.attrs['config'].decode('utf-8')
            
            # Extract weights (will be stored as list of numpy arrays)
            # This path is more reliable than direct load_model for weight extraction
            if 'model_weights' in f:
                weights = []
                # This is a highly generalized way to attempt reading weights from H5 groups
                def collect_weights(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weights.append(np.array(obj))
                f.visititems(collect_weights)
            
        if not config_str:
            raise ValueError("Could not extract JSON config from H5 attributes.")
            
        # The weights collection via visititems is often complex/incorrect. 
        # Let's try the direct Keras load one last time just for weights, as the structure is broken.
        print("Attempting to extract weights via Keras API...")
        weights = load_model(original_path, compile=False).get_weights()
        print(f"Successfully extracted {len(weights)} weight tensors.")

    except Exception as e:
        print(f"FAILED to extract config/weights using low-level methods. Error: {e}")
        # Final desperate attempt to get the JSON config from the error log 
        # (Assuming you are running this after a load failure)
        # If the script itself throws the Keras error, you'd need to manually paste the JSON config here.
        # For now, we rely on the H5 config read being successful above.
        if 'weights' not in locals() or weights is None:
             print("WARNING: Could not reliably extract model weights. Model structure may be saved, but weights will be lost.")
             
    # 2. Perform Aggressive String Replacements on the Config
    try:
        if not config_str:
            raise ValueError("Configuration string is still empty. Aborting.")
        
        print("Applying exhaustive string replacements...")

        # 1. Replace all Keras internal references ('keras.src.')
        config_str = config_str.replace('keras.src.', 'keras.')
        
        # 2. Replace the old Functional module/class names
        config_str = config_str.replace(
            '"module": "keras.models.functional"',
            '"module": "keras.models"'
        )
        config_str = config_str.replace(
            '"class_name": "Functional",',
            '"class_name": "Model",'
        )
        
        # 3. Explicitly target DTypePolicy paths (New Keras vs TF Keras)
        config_str = config_str.replace(
            '"module": "keras.mixed_precision.policy"',
            '"module": "tensorflow.keras.mixed_precision.policy"'
        )
        
        # 4. Remove the shared_object_id if it causes trouble (common issue)
        config_str = config_str.replace(', "shared_object_id": 2182609974928', '')
        config_str = config_str.replace(', "shared_object_id": 2182940816848', '')
        config_str = config_str.replace(', "shared_object_id": 2182940804176', '')
        config_str = config_str.replace(', "shared_object_id": 2182940813072', '')
        
        # 5. Correct the Policy name reference for the mock object
        # NOTE: This is necessary because 'Policy' might be referenced inside config as a DTypePolicy config
        config_str = config_str.replace('"class_name": "DTypePolicy"', '"class_name": "Policy"')


        # 3. Parse the cleaned configuration and re-create the model
        config = json.loads(config_str)

        print("Attempting Model Reconstruction from cleaned JSON config...")
        with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            # This uses the official Keras utility to build the model structure from the cleaned JSON config
            reconstructed_model = model_from_json(config_str, custom_objects=CUSTOM_OBJECTS)
            
        print("Model structure successfully reconstructed.")

        # 4. Apply weights and Save
        if weights and len(weights) > 0:
            reconstructed_model.set_weights(weights)
            print("Model weights successfully applied.")
            
        # Re-save the model in the current environment's compatible format
        reconstructed_model.save(resaved_path)
        print(f"SUCCESS: Model successfully re-saved to: {resaved_path}")

    except Exception as final_e:
        print(f"\n--- FINAL FAILURE ---")
        print(f"FAILED final manual reconstruction for {original_path}. Error: {final_e}")
        print("The serialization error is highly persistent, indicating a severe version mismatch.")
        print("You may need to downgrade your Python/TensorFlow environment to the version the models were originally trained on.")
        
        # Print a portion of the broken config for manual inspection
        if 'config_str' in locals() and config_str:
            print("\n--- Start of Cleaned Config for Manual Debugging ---")
            print(config_str[:2000]) 
            print("--- End of Cleaned Config for Manual Debugging ---\n")
        return


if __name__ == "__main__":
    print("--- Starting Model Resave Process ---")
    
    # 1. Resave the Caption Model
    resave_model_safely(original_caption_path, resaved_caption_path)

    # 2. Resave the Feature Extractor Model
    resave_model_safely(original_feature_path, resaved_feature_path)

    print("\n--- Model Resave Process Complete ---")