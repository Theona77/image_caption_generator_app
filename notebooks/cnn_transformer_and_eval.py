import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
import json
import random
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================================
# FUNGSI HELPER
# ============================================================================

def clean_caption(text):
    """Clean and normalize caption text"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_captions(caption_file):
    """Load and parse captions from TXT file"""
    print(f"Loading captions from {caption_file}...")
    
    try:
        df = pd.read_csv(caption_file, names=["image", "caption"], quotechar='"', sep=",")
        df['caption'] = df['caption'].astype(str).apply(clean_caption)
        df = df[df['caption'].str.len() > 0]
        
        # Group captions by image
        caption_dict = df.groupby("image")["caption"].apply(list).to_dict()
        print(f"✅ Loaded {len(caption_dict)} images with {len(df)} captions")
        
        return caption_dict
    except Exception as e:
        print(f"❌ Error loading captions: {e}")
        return {}

def load_image(img_path, img_size):
    """Load and preprocess image"""
    img = Image.open(img_path).convert("RGB").resize(img_size)
    img_array = np.array(img).astype("float32") / 255.0
    return img_array

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def generate_caption(model, img_path: str, tokenizer, config) -> str:
    """Generate caption using greedy decoding"""
    try:
        # Load image
        img = load_image(img_path, config.IMAGE_SIZE)
        img = np.expand_dims(img, axis=0)
        
        # Get tokens
        start_token = tokenizer.word_index.get('<start>', 1)
        end_token = tokenizer.word_index.get('<end>', 2)
        
        # Initialize caption
        caption_seq = [start_token]
        
        # Generate caption word by word
        for _ in range(config.max_len):
            padded_seq = pad_sequences([caption_seq], maxlen=config.max_len, padding='post')
            predictions = model.predict([img, padded_seq], verbose=0)
            
            current_pos = len(caption_seq) - 1
            if current_pos >= config.max_len:
                break
            
            predicted_id = np.argmax(predictions[0, current_pos])
            
            if predicted_id == end_token:
                break
            
            caption_seq.append(predicted_id)
        
        # Convert to words
        inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
        words = [inv_vocab.get(tid, '<unk>') for tid in caption_seq 
                 if tid not in [0, start_token, end_token]]
        
        return ' '.join(words)
    
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "error generating caption"

def generate_beam_search(model, img_path: str, tokenizer, config) -> str:
    """Generate caption using beam search"""
    try:
        # Load image
        img = load_image(img_path, config.IMAGE_SIZE)
        img = np.expand_dims(img, axis=0)
        
        # Get tokens
        start_token = tokenizer.word_index.get('<start>', 1)
        end_token = tokenizer.word_index.get('<end>', 2)
        beam_width = config.beam_width
        
        # Initialize beams
        beams = [([start_token], 0.0)]
        
        for _ in range(config.max_len):
            new_beams = []
            
            for seq, score in beams:
                if seq[-1] == end_token:
                    new_beams.append((seq, score))
                    continue
                
                padded = pad_sequences([seq], maxlen=config.max_len, padding='post')
                preds = model.predict([img, padded], verbose=0)[0, len(seq)-1]
                
                top_k = np.argsort(preds)[-beam_width:]
                
                for token_id in top_k:
                    new_seq = seq + [token_id]
                    new_score = score - np.log(preds[token_id] + 1e-10)
                    new_beams.append((new_seq, new_score))
            
            beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]
        
        best_seq = beams[0][0]
        
        # Convert to words
        inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
        words = [inv_vocab.get(tid, '<unk>') for tid in best_seq 
                 if tid not in [0, start_token, end_token]]
        
        return ' '.join(words)
    
    except Exception as e:
        print(f"Error in beam search: {e}")
        return "error generating caption"

# ============================================================================
# EVALUATION METRICS CLASSES
# ============================================================================

class BLEUScorer:
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def compute_sentence_bleu(self, reference: List[str], hypothesis: str, n: int = 4) -> float:
        """Menghitung BLEU sentence-level untuk n-gram tertentu."""
        ref_tokens = [ref.split() for ref in reference]
        hyp_tokens = hypothesis.split()
        
        if n == 1:
            weights = (1.0, 0, 0, 0)
        elif n == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n == 3:
            weights = (0.33, 0.33, 0.33, 0)
        else:
            weights = (0.25, 0.25, 0.25, 0.25)
        
        score = sentence_bleu(
            ref_tokens, 
            hyp_tokens, 
            weights=weights,
            smoothing_function=self.smoothing.method1
        )
        return score
    
    def compute_corpus_bleu(self, references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        """Menghitung BLEU corpus-level."""
        refs_tokens = [[ref.split() for ref in refs] for refs in references]
        hyps_tokens = [hyp.split() for hyp in hypotheses]
        
        results = {}
        
        for n, weights in enumerate([(1.0, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)], 1):
            results[f'BLEU-{n}'] = corpus_bleu(
                refs_tokens, hyps_tokens, 
                weights=weights,
                smoothing_function=self.smoothing.method1
            )
            
        return results

class METEORScorer:
    def compute_sentence_meteor(self, reference: List[str], hypothesis: str) -> float:
        """Menghitung METEOR sentence-level."""
        scores = []
        for ref in reference:
            score = meteor_score([ref.split()], hypothesis.split())
            scores.append(score)
        return np.mean(scores)
    
    def compute_corpus_meteor(self, references: List[List[str]], hypotheses: List[str]) -> float:
        """Menghitung METEOR corpus-level."""
        scores = [self.compute_sentence_meteor(refs, hyp) for refs, hyp in zip(references, hypotheses)]
        return np.mean(scores)

class ROUGEScorer:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def compute_sentence_rouge(self, reference: List[str], hypothesis: str) -> Dict[str, float]:
        """Menghitung ROUGE-L sentence-level."""
        scores_list = []
        for ref in reference:
            score = self.scorer.score(ref, hypothesis)
            scores_list.append({
                'precision': score['rougeL'].precision,
                'recall': score['rougeL'].recall,
                'f1': score['rougeL'].fmeasure
            })
        
        return {
            'precision': np.mean([s['precision'] for s in scores_list]),
            'recall': np.mean([s['recall'] for s in scores_list]),
            'f1': np.mean([s['f1'] for s in scores_list])
        }
    
    def compute_corpus_rouge(self, references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        """Menghitung ROUGE-L corpus-level."""
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        for refs, hyp in zip(references, hypotheses):
            scores = self.compute_sentence_rouge(refs, hyp)
            all_precisions.append(scores['precision'])
            all_recalls.append(scores['recall'])
            all_f1s.append(scores['f1'])
        
        return {
            'ROUGE-L-P': np.mean(all_precisions),
            'ROUGE-L-R': np.mean(all_recalls),
            'ROUGE-L-F1': np.mean(all_f1s)
        }

class CIDErScorer:
    def __init__(self, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
    
    def _compute_doc_freq(self, refs_ngrams):
        doc_freq = defaultdict(int)
        for ngrams_dict in refs_ngrams:
            for ngram in ngrams_dict.keys():
                doc_freq[ngram] += 1
        return doc_freq
    
    def _get_ngrams(self, tokens, n):
        ngrams = defaultdict(int)
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                ngrams[ngram] += 1
        return ngrams
    
    def compute_cider(self, references: List[List[str]], hypotheses: List[str]) -> float:
        """Menghitung skor CIDEr."""
        all_refs_ngrams = []
        all_hyps_ngrams = []
        
        for refs, hyp in zip(references, hypotheses):
            ref_tokens_list = [ref.split() for ref in refs]
            hyp_tokens = hyp.split()
            
            refs_ngrams = []
            for ref_tokens in ref_tokens_list:
                ngrams_dict = {}
                for n in range(1, self.n + 1):
                    ngrams_dict.update(self._get_ngrams(ref_tokens, n))
                refs_ngrams.append(ngrams_dict)
            
            hyp_ngrams = {}
            for n in range(1, self.n + 1):
                hyp_ngrams.update(self._get_ngrams(hyp_tokens, n))
                
            all_refs_ngrams.append(refs_ngrams)
            all_hyps_ngrams.append(hyp_ngrams)
        
        all_ngrams_for_doc_freq = []
        for refs_ngrams in all_refs_ngrams:
            all_ngrams_for_doc_freq.extend(refs_ngrams)
        
        doc_freq = self._compute_doc_freq(all_ngrams_for_doc_freq)
        num_docs = len(all_ngrams_for_doc_freq)
        
        scores = []
        
        for refs_ngrams, hyp_ngrams in zip(all_refs_ngrams, all_hyps_ngrams):
            vec_hyp = {}
            vec_refs = []
            
            if hyp_ngrams:
                len_hyp_ngrams = sum(hyp_ngrams.values()) 
                for ngram, count in hyp_ngrams.items():
                    tf = count / len_hyp_ngrams
                    idf = np.log((num_docs + 1) / (doc_freq.get(ngram, 0) + 1))
                    vec_hyp[ngram] = tf * idf
            
            for ref_ngrams in refs_ngrams:
                vec_ref = {}
                if ref_ngrams:
                    len_ref_ngrams = sum(ref_ngrams.values())
                    for ngram, count in ref_ngrams.items():
                        tf = count / len_ref_ngrams
                        idf = np.log((num_docs + 1) / (doc_freq.get(ngram, 0) + 1))
                        vec_ref[ngram] = tf * idf
                vec_refs.append(vec_ref)
            
            similarities = []
            for vec_ref in vec_refs:
                all_ngrams = set(vec_hyp.keys()) | set(vec_ref.keys())
                
                dot_product = sum(vec_hyp.get(k, 0) * vec_ref.get(k, 0) for k in all_ngrams)
                norm_hyp = np.sqrt(sum(vec_hyp.get(k, 0)**2 for k in all_ngrams))
                norm_ref = np.sqrt(sum(vec_ref.get(k, 0)**2 for k in all_ngrams))
                
                if norm_hyp > 0 and norm_ref > 0:
                    sim = dot_product / (norm_hyp * norm_ref)
                else:
                    sim = 0.0
                similarities.append(sim)
            
            score = np.mean(similarities) * 10.0
            scores.append(score)
            
        return np.mean(scores) if scores else 0.0

class CaptionEvaluator:
    def __init__(self):
        self.bleu_scorer = BLEUScorer()
        self.meteor_scorer = METEORScorer()
        self.rouge_scorer = ROUGEScorer()
        self.cider_scorer = CIDErScorer()
    
    def evaluate(self, references: List[List[str]], hypotheses: List[str], 
                 verbose: bool = True) -> Dict[str, float]:
        """Evaluasi korpus caption yang dihasilkan terhadap referensi."""
        if len(references) != len(hypotheses):
            raise ValueError("Jumlah referensi dan hipotesis harus sama")
        
        results = {}
        
        if verbose: print("Computing BLEU scores...")
        results.update(self.bleu_scorer.compute_corpus_bleu(references, hypotheses))
        
        if verbose: print("Computing METEOR score...")
        results['METEOR'] = self.meteor_scorer.compute_corpus_meteor(references, hypotheses)
        
        if verbose: print("Computing ROUGE-L scores...")
        results.update(self.rouge_scorer.compute_corpus_rouge(references, hypotheses))
        
        if verbose: print("Computing CIDEr score...")
        results['CIDEr'] = self.cider_scorer.compute_cider(references, hypotheses)
        
        # Average score
        metric_values = [
            results['BLEU-1'], results['BLEU-2'], results['BLEU-3'], results['BLEU-4'],
            results['METEOR'], results['ROUGE-L-F1'], results['CIDEr']
        ]
        results['AVG_SCORE'] = np.mean(metric_values)
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Number of samples: {len(hypotheses)}")
            print()
            print("BLEU Scores:")
            print(f"  BLEU-1:  {results['BLEU-1']:.4f}")
            print(f"  BLEU-2:  {results['BLEU-2']:.4f}")
            print(f"  BLEU-3:  {results['BLEU-3']:.4f}")
            print(f"  BLEU-4:  {results['BLEU-4']:.4f}")
            print()
            print(f"METEOR:    {results['METEOR']:.4f}")
            print()
            print("ROUGE-L Scores:")
            print(f"  Precision: {results['ROUGE-L-P']:.4f}")
            print(f"  Recall:    {results['ROUGE-L-R']:.4f}")
            print(f"  F1-Score:  {results['ROUGE-L-F1']:.4f}")
            print()
            print(f"CIDEr:     {results['CIDEr']:.4f}")
            print("-" * 60)
            print(f"AVERAGE AGGREGATE SCORE: {results['AVG_SCORE']:.4f}")
            print("="*60)
        
        return results
    
    def evaluate_single(self, references: List[str], hypothesis: str) -> Dict[str, float]:
        """Evaluasi satu caption."""
        results = {}
        
        for n in [1, 2, 3, 4]:
            results[f'BLEU-{n}'] = self.bleu_scorer.compute_sentence_bleu(references, hypothesis, n)
        
        results['METEOR'] = self.meteor_scorer.compute_sentence_meteor(references, hypothesis)
        
        rouge = self.rouge_scorer.compute_sentence_rouge(references, hypothesis)
        results['ROUGE-L-F1'] = rouge['f1']
        
        return results

# ============================================================================
# MODEL LOADING WITH CUSTOM OBJECTS
# ============================================================================

def create_look_ahead_mask(size):
    """Create causal mask for decoder self-attention"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def build_model_from_config(config, vocab_size):
    """Rebuild model architecture untuk loading weights"""
    from tensorflow.keras import layers, models
    
    print("   Building CNN Encoder...")
    # CNN Encoder
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*config.IMAGE_SIZE, 3),
        pooling=None
    )
    base_model.trainable = True
    
    inp = layers.Input(shape=(*config.IMAGE_SIZE, 3), name='image_input')
    x = base_model(inp)
    
    shape = tf.shape(x)
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((h * w, c))(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    
    encoder = models.Model(inp, x, name='cnn_encoder')
    
    print("   Building Transformer Decoder...")
    # Transformer Decoder
    img_features = layers.Input(shape=(None, 256), name='img_features')
    caption_input = layers.Input(shape=(config.max_len,), dtype='int32', name='caption_input')
    
    token_emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=256,
        mask_zero=True,
        name='token_embedding'
    )(caption_input)
    
    positions = tf.range(start=0, limit=config.max_len, delta=1)
    pos_emb = layers.Embedding(
        input_dim=config.max_len,
        output_dim=256,
        name='position_embedding'
    )(positions)
    
    x = token_emb + pos_emb
    x = layers.Dropout(0.1)(x)
    
    causal_mask = create_look_ahead_mask(config.max_len)
    causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]
    
    # Transformer blocks
    for i in range(2):  # num_layers = 2
        # Self-attention
        attn1 = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=256,
            dropout=0.1
        )(query=x, value=x, key=x, attention_mask=causal_mask)
        x = layers.LayerNormalization()(x + attn1)
        
        # Cross-attention
        attn2 = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=256,
            dropout=0.1
        )(query=x, value=img_features, key=img_features)
        x = layers.LayerNormalization()(x + attn2)
        
        # FFN
        ffn = layers.Dense(512, activation='relu')(x)
        ffn = layers.Dropout(0.1)(ffn)
        ffn = layers.Dense(256)(ffn)
        x = layers.LayerNormalization()(x + ffn)
    
    outputs = layers.Dense(vocab_size, activation='softmax', name='output')(x)
    decoder = models.Model([img_features, caption_input], outputs, name='transformer_decoder')
    
    # Full model
    img_input = layers.Input(shape=(*config.IMAGE_SIZE, 3), name='image')
    cap_input = layers.Input(shape=(config.max_len,), dtype='int32', name='caption')
    
    img_features_out = encoder(img_input)
    outputs_final = decoder([img_features_out, cap_input])
    
    model = models.Model([img_input, cap_input], outputs_final, name='image_captioning')
    
    return model

def load_model_safe(model_path, config, vocab_size):
    """Safe model loading dengan fallback options"""
    
    # Method 1: Load dengan custom objects
    try:
        print("   Method 1: Loading with custom objects...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'MultiHeadAttention': tf.keras.layers.MultiHeadAttention},
            compile=False
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        print("   ✅ Success!")
        return model
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 2: Rebuild dan load weights
    try:
        print("   Method 2: Rebuilding model and loading weights...")
        model = build_model_from_config(config, vocab_size)
        model.load_weights(model_path)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        print("   ✅ Success!")
        return model
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 3: Load as SavedModel format
    try:
        print("   Method 3: Loading as SavedModel format...")
        model = tf.saved_model.load(model_path.replace('.h5', ''))
        print("   ✅ Success!")
        return model
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    return None

# ============================================================================
# MAIN EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model, test_images: List[str], caption_dict: Dict[str, List[str]], 
                   img_folder: str, tokenizer, config, 
                   use_beam_search: bool = False) -> Tuple[Dict[str, float], List[List[str]], List[str]]:
    """Evaluasi model pada set pengujian."""
    
    print(f"Evaluating on {len(test_images)} test images...")
    
    references = []
    hypotheses = []
    
    for i, img_name in enumerate(test_images):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(test_images)}")
        
        img_path = os.path.join(img_folder, img_name)
        
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Warning: Image not found - {img_name}")
            continue
        
        refs = caption_dict.get(img_name, [])
        if not refs:
            continue
        
        references.append(refs)
        
        try:
            if use_beam_search:
                hyp = generate_beam_search(model, img_path, tokenizer, config)
            else:
                hyp = generate_caption(model, img_path, tokenizer, config)
            hypotheses.append(hyp)
        except Exception as e:
            print(f"Error generating caption for {img_name}: {e}")
            hypotheses.append("")
    
    evaluator = CaptionEvaluator()
    results = evaluator.evaluate(references, hypotheses, verbose=True)
    
    return results, references, hypotheses

def analyze_predictions(references: List[List[str]], hypotheses: List[str], 
                        image_names: List[str] = None, n_samples: int = 10):
    """Menampilkan contoh kualitatif prediksi."""
    evaluator = CaptionEvaluator()
    
    print("\n" + "="*80)
    print("QUALITATIVE ANALYSIS")
    print("="*80)
    
    indices = np.random.choice(len(hypotheses), min(n_samples, len(hypotheses)), replace=False)
    
    for idx in indices:
        refs = references[idx]
        hyp = hypotheses[idx]
        
        scores = evaluator.evaluate_single(refs, hyp)
        
        print(f"\nSample {idx+1}")
        if image_names:
            print(f"Image: {image_names[idx]}")
        print("-" * 80)
        print("Reference Captions:")
        for i, ref in enumerate(refs, 1):
            print(f"  {i}. {ref}")
        print(f"\nGenerated Caption:")
        print(f"  → {hyp}")
        print(f"\nScores:")
        print(f"  BLEU-4: {scores['BLEU-4']:.4f}")
        print(f"  METEOR: {scores['METEOR']:.4f}")
        print(f"  ROUGE-L: {scores['ROUGE-L-F1']:.4f}")
        print("=" * 80)

def save_evaluation_results(results: Dict[str, float], references: List[List[str]], 
                            hypotheses: List[str], image_names: List[str] = None,
                            output_dir: str = "evaluation_results"):
    """Menyimpan hasil evaluasi ke file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    predictions_data = []
    for i, (refs, hyp) in enumerate(zip(references, hypotheses)):
        entry = {
            'id': i,
            'references': refs,
            'hypothesis': hyp
        }
        if image_names:
            entry['image'] = image_names[i]
        predictions_data.append(entry)
    
    with open(os.path.join(output_dir, "predictions.json"), 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    # Save to CSV
    df = pd.DataFrame({
        'image': image_names if image_names else range(len(hypotheses)),
        'generated': hypotheses,
        'reference_1': [refs[0] if len(refs) > 0 else "" for refs in references],
        'reference_2': [refs[1] if len(refs) > 1 else "" for refs in references],
    })
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}/")

def main_evaluation(model_path: str, test_images: List[str], 
                    caption_dict: Dict[str, List[str]], img_folder: str,
                    tokenizer_path: str, config, use_beam_search: bool = True):
    """Pipeline evaluasi lengkap."""
    
    print("\n" + "="*80)
    print("LOADING MODEL AND TOKENIZER")
    print("="*80)
    
    # Load tokenizer first (to get vocab_size)
    try:
        print(f"Loading tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        vocab_size = min(10000, len(tokenizer.word_index) + 1)
        print(f"✅ Tokenizer loaded successfully (vocab_size: {vocab_size})")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None
    
    # Load model with safe method
    print(f"\nLoading model from {model_path}...")
    model = load_model_safe(model_path, config, vocab_size)
    
    if model is None:
        print("\n❌ All model loading methods failed!")
        print("💡 Suggestions:")
        print("   1. Check if model file exists and is not corrupted")
        print("   2. Try re-saving the model using: model.save('model.h5')")
        print("   3. Save in SavedModel format: model.save('model_dir/')")
        return None
    
    print("✅ Model loaded successfully!")
    
    # Run evaluation
    results, references, hypotheses = evaluate_model(
        model, test_images, caption_dict, img_folder, 
        tokenizer, config, use_beam_search
    )
    
    # Analyze predictions
    analyze_predictions(references, hypotheses, test_images, n_samples=5)
    
    # Save results
    save_evaluation_results(results, references, hypotheses, test_images)
    
    return results

# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "#" * 80)
    print("IMAGE CAPTIONING MODEL EVALUATION")
    print("#" * 80)
    
    # 1. Setup Configuration
    class ModelConfig:
        def __init__(self):
            self.max_len = 40  
            self.beam_width = 3
            self.IMAGE_SIZE = (299, 299)
    
    config = ModelConfig()
    
    # 2. Load caption dictionary
    print("\n📂 Loading caption dictionary...")
    try:
        caption_dict = load_captions("captions.txt")
        ALL_IMAGE_NAMES = list(caption_dict.keys())
        
        if len(ALL_IMAGE_NAMES) == 0:
            print("❌ No captions loaded. Exiting...")
            exit(1)
            
        print(f"✅ Total images available: {len(ALL_IMAGE_NAMES)}")
    except Exception as e:
        print(f"❌ Error loading captions: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 3. Define number of test samples
    N_TEST_SAMPLES = 100  
    
    # 4. Sample images randomly
    if len(ALL_IMAGE_NAMES) > N_TEST_SAMPLES:
        sampled_image_names = random.sample(ALL_IMAGE_NAMES, N_TEST_SAMPLES)
        print(f"📊 Sampling {N_TEST_SAMPLES} images from {len(ALL_IMAGE_NAMES)} total")
    else:
        sampled_image_names = ALL_IMAGE_NAMES
        print(f"📊 Using all {len(ALL_IMAGE_NAMES)} images (less than {N_TEST_SAMPLES})")
    
    # 5. Create caption dictionary for sampled images
    sampled_caption_dict = {
        img: caption_dict[img] for img in sampled_image_names
    }
    
    print(f"\n🚀 Starting evaluation on {len(sampled_image_names)} images")
    print(f"   Inference method: {'Beam Search' if True else 'Greedy'} (beam_width={config.beam_width})")
    
    # 6. Run main evaluation pipeline
    try:
        final_results = main_evaluation(
            model_path="saved_models/best_model.h5",
            test_images=sampled_image_names,
            caption_dict=sampled_caption_dict,
            img_folder="Images",
            tokenizer_path="saved_models/tokenizer.pkl",
            config=config,
            use_beam_search=True
        )
        
        if final_results is None:
            print("\n" + "=" * 80)
            print("❌ EVALUATION FAILED")
            print("=" * 80)
            print("\n💡 Troubleshooting:")
            print("1. Model Loading Issues:")
            print("   - Check if 'saved_models/best_model.h5' exists")
            print("   - Try re-training and saving model again")
            print("   - Use SavedModel format: model.save('saved_models/model_dir/')")
            print("\n2. Alternative: Re-save your model using this code:")
            print("   ```python")
            print("   # Load your trained model")
            print("   model = your_trained_model")
            print("   # Save in h5 format")
            print("   model.save('saved_models/best_model.h5', save_format='h5')")
            print("   # OR save in SavedModel format")
            print("   model.save('saved_models/model_dir/')")
            print("   ```")
            exit(1)
        
        print("\n" + "=" * 80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # 7. Display final results
        print("\n📊 FINAL EVALUATION SUMMARY:")
        print("-" * 80)
        print(f"Images evaluated    : {len(sampled_image_names)}")
        print(f"\n📈 BLEU Scores:")
        print(f"  BLEU-1  : {final_results.get('BLEU-1', 0):.4f}")
        print(f"  BLEU-2  : {final_results.get('BLEU-2', 0):.4f}")
        print(f"  BLEU-3  : {final_results.get('BLEU-3', 0):.4f}")
        print(f"  BLEU-4  : {final_results.get('BLEU-4', 0):.4f}")
        print(f"\n📈 Other Metrics:")
        print(f"  METEOR  : {final_results.get('METEOR', 0):.4f}")
        print(f"  ROUGE-L : {final_results.get('ROUGE-L-F1', 0):.4f}")
        print(f"  CIDEr   : {final_results.get('CIDEr', 0):.4f}")
        print(f"\n🎯 Average Score: {final_results.get('AVG_SCORE', 0):.4f}")
        print("-" * 80)
        
        # 8. Save to JSON
        output_file = "evaluation_summary.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\n💾 Results saved to: {output_file}")
        print(f"💾 Detailed results in: evaluation_results/")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        print("\n🔍 Required files:")
        print("  ✓ saved_models/best_model.h5")
        print("  ✓ saved_models/tokenizer.pkl")
        print("  ✓ captions.txt")
        print("  ✓ Images/ (folder with images)")
        
    except Exception as e:
        print(f"\n❌ Unexpected error during evaluation:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()