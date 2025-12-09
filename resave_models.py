import os
from tensorflow.keras.models import load_model

# --- Configuration ---
# IMPORTANT: Update these paths to point to your *original* saved model files.
MODEL_DIR = 'outputs/models'
ORIGINAL_CAPTION_MODEL_NAME = 'model.keras'
ORIGINAL_FEATURE_EXTRACTOR_NAME = 'feature_extractor.keras'

# New names for the re-saved, fixed models
RESAVED_CAPTION_MODEL_NAME = 'model_resaved.keras'
RESAVED_FEATURE_EXTRACTOR_NAME = 'feature_extractor_resaved.keras'

# --- Paths ---
original_caption_path = os.path.join(MODEL_DIR, ORIGINAL_CAPTION_MODEL_NAME)
original_feature_path = os.path.join(MODEL_DIR, ORIGINAL_FEATURE_EXTRACTOR_NAME)

resaved_caption_path = os.path.join(MODEL_DIR, RESAVED_CAPTION_MODEL_NAME)
resaved_feature_path = os.path.join(MODEL_DIR, RESAVED_FEATURE_EXTRACTOR_NAME)


def resave_model_safely(original_path, resaved_path):
    """
    Loads an older Keras model with compile=False to bypass configuration errors,
    then saves it in the new format to fix internal layer arguments.
    """
    if not os.path.exists(original_path):
        print(f"ERROR: Original model not found at {original_path}")
        return

    try:
        # Load the model with compile=False to bypass compilation/config errors
        print(f"Attempting to load: {original_path}")
        old_model = load_model(original_path, compile=False)
        print("Model loaded successfully.")

        # Re-save the model in the new Keras format
        old_model.save(resaved_path)
        print(f"Model successfully re-saved to: {resaved_path}")

    except Exception as e:
        print(f"FAILED to process model at {original_path}. Error: {e}")


if __name__ == "__main__":
    print("--- Starting Model Resave Process ---")
    
    # 1. Resave the Caption Model
    resave_model_safely(original_caption_path, resaved_caption_path)

    # 2. Resave the Feature Extractor Model
    resave_model_safely(original_feature_path, resaved_feature_path)

    print("--- Model Resave Process Complete ---")