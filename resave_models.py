from tensorflow.keras.models import load_model

# Load your original models
caption_model = load_model('outputs/models/model.keras', compile=False)
feature_extractor = load_model('outputs/models/feature_extractor.keras', compile=False)

# Re-save them to ensure compatibility
caption_model.save('outputs/models/model_fixed.keras')
feature_extractor.save('outputs/models/feature_extractor_fixed.keras')
