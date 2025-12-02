#!/usr/bin/env python
# coding: utf-8

# #### Import library

# In[128]:


import os
import re
import random
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# #### Settings

# In[129]:


class Config:
    # Paths
    CAPTION_FILE = "captions.txt"  
    IMAGE_FOLDER = "Images"  
    SAVE_DIR = "saved_models"
    
    # Image settings
    IMAGE_SIZE = (299, 299)  
    
    # Model hyperparameters
    EMBED_DIM = 256
    NUM_HEADS = 4
    FF_DIM = 512
    NUM_LAYERS = 2
    
    # Training settings
    MAX_LEN = 40  
    VOCAB_SIZE = 10000
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

config = Config()

# Create directories
os.makedirs(config.SAVE_DIR, exist_ok=True)


# #### Data loading & Preprocessing

# In[130]:


def clean_caption(text):
    """Clean and normalize caption text"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def load_captions(caption_file):
    """Load and parse captions file"""
    print("Loading captions...")
    df = pd.read_csv(caption_file, names=["image", "caption"])
    df['caption'] = df['caption'].astype(str).apply(clean_caption)
    df = df[df['caption'].str.len() > 0]  # Remove empty captions
    
    # Group captions by image
    caption_dict = df.groupby("image")["caption"].apply(list).to_dict()
    print(f"Loaded {len(caption_dict)} images with {len(df)} captions")
    
    return caption_dict


# #### Tokenizer Setup

# In[131]:


def create_tokenizer(caption_dict, vocab_size):
    """Create tokenizer with special tokens"""
    print("Creating tokenizer...")
    
    # Prepare all captions with special tokens
    all_captions = []
    for captions in caption_dict.values():
        for cap in captions:
            text = f"<start> {cap} <end>"
            all_captions.append(text)
    
    # Fit tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>", filters='')
    tokenizer.fit_on_texts(all_captions)
    
    vocab_size = min(vocab_size, len(tokenizer.word_index) + 1)
    print(f"Vocabulary size: {vocab_size}")
    
    return tokenizer, vocab_size


# #### Data Preparation

# In[132]:


def load_image(img_path, img_size):
    """Load and preprocess image"""
    img = Image.open(img_path).convert("RGB").resize(img_size)
    img_array = np.array(img).astype("float32") / 255.0
    return img_array

def caption_to_sequences(caption, tokenizer, max_len):
    """Convert caption to input and output sequences"""
    text = f"<start> {caption} <end>"
    seq = tokenizer.texts_to_sequences([text])[0]
    
    # Input: without last token, Output: without first token
    seq_in = seq[:-1]
    seq_out = seq[1:]
    
    # Pad sequences
    seq_in = pad_sequences([seq_in], maxlen=max_len, padding='post')[0]
    seq_out = pad_sequences([seq_out], maxlen=max_len, padding='post')[0]
    
    return seq_in, seq_out


# #### Data Generator

# In[133]:


class DataGenerator(tf.keras.utils.Sequence):
    """Efficient data generator for training"""
    
    def __init__(self, image_list, caption_dict, img_folder, tokenizer, 
                 batch_size, max_len, img_size, shuffle=True):
        self.image_list = image_list
        self.caption_dict = caption_dict
        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_list) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_list[k] for k in indices]
        return self._generate_batch(batch_images)
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _generate_batch(self, batch_images):
        batch_imgs = []
        batch_cap_in = []
        batch_cap_out = []
        
        for img_name in batch_images:
            img_path = os.path.join(self.img_folder, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            try:
                # Load image
                img = load_image(img_path, self.img_size)
                
                # Random caption for variety
                caption = random.choice(self.caption_dict[img_name])
                seq_in, seq_out = caption_to_sequences(caption, self.tokenizer, self.max_len)
                
                batch_imgs.append(img)
                batch_cap_in.append(seq_in)
                batch_cap_out.append(seq_out)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                continue
        
        if len(batch_imgs) == 0:
            # Return empty batch (shouldn't happen often)
            return [np.zeros((1, *self.img_size, 3)), 
                    np.zeros((1, self.max_len))], np.zeros((1, self.max_len))
        
        return [np.array(batch_imgs), np.array(batch_cap_in)], np.array(batch_cap_out)


# #### Model Architecture

# In[134]:


def create_look_ahead_mask(size):
    """Create causal mask for decoder self-attention"""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


# In[135]:


def build_cnn_encoder(img_size, embed_dim):
    """CNN Encoder using EfficientNetB0 with spatial features"""
    print("Building CNN Encoder...")
    
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*img_size, 3),
        pooling=None  # Keep spatial features
    )
    base_model.trainable = False  # Freeze for faster training
    
    inp = layers.Input(shape=(*img_size, 3), name='image_input')
    x = base_model(inp)  # (batch, H, W, C)
    
    # Reshape to sequence
    shape = tf.shape(x)
    batch_size = shape[0]
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    
    x = layers.Reshape((h * w, c))(x)  # (batch, H*W, C)
    x = layers.Dense(embed_dim, activation='relu')(x)
    x = layers.LayerNormalization()(x)
    
    model = models.Model(inp, x, name='cnn_encoder')
    return model


# In[136]:


def transformer_decoder_block(x, img_features, embed_dim, num_heads, ff_dim, 
                               causal_mask, dropout=0.1):
    """Single Transformer Decoder Block"""
    
    # 1. Self-attention with causal mask
    attn1 = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        dropout=dropout
    )(query=x, value=x, key=x, attention_mask=causal_mask)
    x = layers.LayerNormalization()(x + attn1)
    
    # 2. Cross-attention to image features
    attn2 = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        dropout=dropout
    )(query=x, value=img_features, key=img_features)
    x = layers.LayerNormalization()(x + attn2)
    
    # 3. Feed-forward network
    ffn = layers.Dense(ff_dim, activation='relu')(x)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(embed_dim)(ffn)
    x = layers.LayerNormalization()(x + ffn)
    
    return x


# In[137]:


def build_transformer_decoder(vocab_size, embed_dim, num_heads, ff_dim, 
                               num_layers, max_len):
    """Transformer Decoder with causal masking"""
    print("Building Transformer Decoder...")
    
    # Inputs
    img_features = layers.Input(shape=(None, embed_dim), name='img_features')
    caption_input = layers.Input(shape=(max_len,), dtype='int32', name='caption_input')
    
    # Token embeddings
    token_emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        name='token_embedding'
    )(caption_input)
    
    # Positional embeddings
    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_emb = layers.Embedding(
        input_dim=max_len,
        output_dim=embed_dim,
        name='position_embedding'
    )(positions)
    
    x = token_emb + pos_emb
    x = layers.Dropout(0.1)(x)
    
    # Create causal mask
    causal_mask = create_look_ahead_mask(max_len)
    
    # Stack transformer blocks
    for i in range(num_layers):
        x = transformer_decoder_block(
            x, img_features, embed_dim, num_heads, ff_dim, 
            causal_mask, dropout=0.1
        )
    
    # Output layer
    outputs = layers.Dense(vocab_size, activation='softmax', name='output')(x)
    
    model = models.Model([img_features, caption_input], outputs, name='transformer_decoder')
    return model


# In[138]:


def build_complete_model(config, vocab_size):
    """Build complete end-to-end model"""
    print("\nBuilding complete model...")
    
    # Encoder
    encoder = build_cnn_encoder(config.IMAGE_SIZE, config.EMBED_DIM)
    
    # Decoder
    decoder = build_transformer_decoder(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        ff_dim=config.FF_DIM,
        num_layers=config.NUM_LAYERS,
        max_len=config.MAX_LEN
    )
    
    # Full model
    img_input = layers.Input(shape=(*config.IMAGE_SIZE, 3), name='image')
    cap_input = layers.Input(shape=(config.MAX_LEN,), dtype='int32', name='caption')
    
    img_features = encoder(img_input)
    outputs = decoder([img_features, cap_input])
    
    model = models.Model([img_input, cap_input], outputs, name='image_captioning')
    
    return model, encoder, decoder


# In[139]:


def train_model(model, train_gen, val_gen, config):
    """Train the model with callbacks"""
    print("\nCompiling model...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(config.SAVE_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# #### Inference

# In[140]:


def generate_caption(model, image_path, tokenizer, config, max_len=None):
    """Generate caption for a single image using greedy decoding"""
    if max_len is None:
        max_len = config.MAX_LEN
    
    # Load and preprocess image
    img = load_image(image_path, config.IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    
    # Get special tokens
    start_token = tokenizer.word_index.get('<start>', 1)
    end_token = tokenizer.word_index.get('<end>', 2)
    
    # Initialize caption with start token
    caption_seq = [start_token]
    
    # Generate caption word by word
    for _ in range(max_len):
        # Pad current sequence
        padded_seq = pad_sequences([caption_seq], maxlen=max_len, padding='post')
        
        # Predict next token
        predictions = model.predict([img, padded_seq], verbose=0)
        
        # Get prediction at current position
        current_pos = len(caption_seq) - 1
        if current_pos >= max_len:
            break
        
        predicted_id = np.argmax(predictions[0, current_pos])
        
        # Stop if end token
        if predicted_id == end_token:
            break
        
        caption_seq.append(predicted_id)
    
    # Convert tokens to words
    inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
    caption_words = []
    
    for token_id in caption_seq:
        if token_id == 0:  # padding
            continue
        word = inv_vocab.get(token_id, '<unk>')
        if word in ['<start>', '<end>']:
            continue
        caption_words.append(word)
    
    return ' '.join(caption_words)


# In[141]:


def generate_beam_search(model, image_path, tokenizer, config, beam_width=3):
    """Generate caption using beam search (better quality)"""
    img = load_image(image_path, config.IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    
    start_token = tokenizer.word_index.get('<start>', 1)
    end_token = tokenizer.word_index.get('<end>', 2)
    
    # Initialize beam with start token
    beams = [([start_token], 0.0)]  # (sequence, score)
    
    for _ in range(config.MAX_LEN):
        new_beams = []
        
        for seq, score in beams:
            if seq[-1] == end_token:
                new_beams.append((seq, score))
                continue
            
            # Pad and predict
            padded = pad_sequences([seq], maxlen=config.MAX_LEN, padding='post')
            preds = model.predict([img, padded], verbose=0)[0, len(seq)-1]
            
            # Get top k predictions
            top_k = np.argsort(preds)[-beam_width:]
            
            for token_id in top_k:
                new_seq = seq + [token_id]
                new_score = score - np.log(preds[token_id] + 1e-10)
                new_beams.append((new_seq, new_score))
        
        # Keep top beam_width beams
        beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]
    
    # Get best sequence
    best_seq = beams[0][0]
    
    # Convert to words
    inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
    words = [inv_vocab.get(tid, '<unk>') for tid in best_seq 
             if tid not in [0, start_token, end_token]]
    
    return ' '.join(words)


# In[142]:


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("IMAGE CAPTIONING: CNN + TRANSFORMER")
    print("=" * 70)
    
    # 1. Load data
    caption_dict = load_captions(config.CAPTION_FILE)
    
    # 2. Create tokenizer
    tokenizer, vocab_size = create_tokenizer(caption_dict, config.VOCAB_SIZE)
    
    # Save tokenizer
    import pickle
    with open(os.path.join(config.SAVE_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # 3. Split data
    all_images = list(caption_dict.keys())
    random.shuffle(all_images)
    
    train_size = int(len(all_images) * config.TRAIN_SPLIT)
    val_size = int(len(all_images) * config.VAL_SPLIT)
    
    train_images = all_images[:train_size]
    val_images = all_images[train_size:train_size + val_size]
    test_images = all_images[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)}")
    print(f"  Val:   {len(val_images)}")
    print(f"  Test:  {len(test_images)}")
    
    # 4. Create generators
    train_gen = DataGenerator(
        train_images, caption_dict, config.IMAGE_FOLDER,
        tokenizer, config.BATCH_SIZE, config.MAX_LEN, config.IMAGE_SIZE
    )
    
    val_gen = DataGenerator(
        val_images, caption_dict, config.IMAGE_FOLDER,
        tokenizer, config.BATCH_SIZE, config.MAX_LEN, config.IMAGE_SIZE,
        shuffle=False
    )
    
    # 5. Build model
    model, encoder, decoder = build_complete_model(config, vocab_size)
    model.summary()
    
    # 6. Train
    history = train_model(model, train_gen, val_gen, config)
    
    # 7. Save final model
    model.save(os.path.join(config.SAVE_DIR, 'final_model.h5'))
    print(f"\nModel saved to {config.SAVE_DIR}")
    
    # 8. Test inference
    if len(test_images) > 0:
        test_img = os.path.join(config.IMAGE_FOLDER, test_images[0])
        print(f"\nTest inference on: {test_images[0]}")
        caption = generate_caption(model, test_img, tokenizer, config)
        print(f"Generated caption: {caption}")
    
    return model, tokenizer, history


# In[143]:


def load_and_predict(image_path, use_beam_search=False):
    """Load saved model and generate caption"""
    import pickle
    
    # Load tokenizer
    with open(os.path.join(config.SAVE_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load model
    model = tf.keras.models.load_model(os.path.join(config.SAVE_DIR, 'best_model.h5'))
    
    # Generate caption
    if use_beam_search:
        caption = generate_beam_search(model, image_path, tokenizer, config)
    else:
        caption = generate_caption(model, image_path, tokenizer, config)
    
    return caption


# In[144]:


if __name__ == "__main__":
    # For training:
    model, tokenizer, history = main()


# In[145]:


import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
import re
import json


# In[146]:


import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# In[147]:


class BLEUScorer:
    
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def compute_sentence_bleu(self, reference: List[str], hypothesis: str, n: int = 4) -> float:
        # Tokenize
        ref_tokens = [ref.split() for ref in reference]
        hyp_tokens = hypothesis.split()
        
        # Set weights based on n
        if n == 1:
            weights = (1.0, 0, 0, 0)
        elif n == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n == 3:
            weights = (0.33, 0.33, 0.33, 0)
        else:  # n == 4
            weights = (0.25, 0.25, 0.25, 0.25)
        
        # Compute BLEU
        score = sentence_bleu(
            ref_tokens, 
            hyp_tokens, 
            weights=weights,
            smoothing_function=self.smoothing.method1
        )
        
        return score
    
    def compute_corpus_bleu(self, references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        # Tokenize all
        refs_tokens = [[ref.split() for ref in refs] for refs in references]
        hyps_tokens = [hyp.split() for hyp in hypotheses]
        
        results = {}
        
        # BLEU-1
        results['BLEU-1'] = corpus_bleu(
            refs_tokens, hyps_tokens, 
            weights=(1.0, 0, 0, 0),
            smoothing_function=self.smoothing.method1
        )
        
        # BLEU-2
        results['BLEU-2'] = corpus_bleu(
            refs_tokens, hyps_tokens, 
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=self.smoothing.method1
        )
        
        # BLEU-3
        results['BLEU-3'] = corpus_bleu(
            refs_tokens, hyps_tokens, 
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=self.smoothing.method1
        )
        
        # BLEU-4
        results['BLEU-4'] = corpus_bleu(
            refs_tokens, hyps_tokens, 
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing.method1
        )
        
        return results


# In[148]:


class METEORScorer:
    
    def compute_sentence_meteor(self, reference: List[str], hypothesis: str) -> float:
        """Compute METEOR for single sentence"""
        # METEOR expects single reference and hypothesis as strings
        # We'll average over multiple references
        scores = []
        for ref in reference:
            score = meteor_score([ref.split()], hypothesis.split())
            scores.append(score)
        
        return np.mean(scores)
    
    def compute_corpus_meteor(self, references: List[List[str]], hypotheses: List[str]) -> float:
        """Compute METEOR for entire corpus"""
        scores = []
        
        for refs, hyp in zip(references, hypotheses):
            score = self.compute_sentence_meteor(refs, hyp)
            scores.append(score)
        
        return np.mean(scores)


# In[149]:


class ROUGEScorer:
    """
    ROUGE-L: Measures longest common subsequence
    Focus on recall rather than precision
    
    Range: 0-1 (higher is better)
    Returns: Precision, Recall, F1-score
    Good F1 score: > 0.4
    """
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def compute_sentence_rouge(self, reference: List[str], hypothesis: str) -> Dict[str, float]:
        """Compute ROUGE-L for single sentence"""
        scores_list = []
        
        for ref in reference:
            score = self.scorer.score(ref, hypothesis)
            scores_list.append({
                'precision': score['rougeL'].precision,
                'recall': score['rougeL'].recall,
                'f1': score['rougeL'].fmeasure
            })
        
        # Average over references
        avg_scores = {
            'precision': np.mean([s['precision'] for s in scores_list]),
            'recall': np.mean([s['recall'] for s in scores_list]),
            'f1': np.mean([s['f1'] for s in scores_list])
        }
        
        return avg_scores
    
    def compute_corpus_rouge(self, references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        """Compute ROUGE-L for entire corpus"""
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


# In[150]:


class CIDErScorer:
    """
    CIDEr: Specialized metric for image captioning
    Measures consensus between generated caption and human captions
    Uses TF-IDF weighting
    
    Range: 0-10+ (higher is better)
    Good score: > 0.8
    """
    
    def __init__(self, n=4, sigma=6.0):
        """
        Args:
            n: max n-gram order
            sigma: standard deviation for Gaussian penalty
        """
        self.n = n
        self.sigma = sigma
    
    def _compute_doc_freq(self, refs_ngrams):
        """Compute document frequency for IDF"""
        doc_freq = defaultdict(int)
        
        for ngrams_dict in refs_ngrams:
            for ngram in ngrams_dict.keys():
                doc_freq[ngram] += 1
        
        return doc_freq
    
    def _get_ngrams(self, tokens, n):
        """Get n-grams from tokens"""
        ngrams = defaultdict(int)
        
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        
        return ngrams
    
    def compute_cider(self, references: List[List[str]], hypotheses: List[str]) -> float:
        """
        Compute CIDEr score
        
        This is a simplified implementation. For exact CIDEr scores,
        use the official pycocoevalcap package.
        """
        scores = []
        
        # Get all n-grams
        all_refs_ngrams = []
        all_hyps_ngrams = []
        
        for refs, hyp in zip(references, hypotheses):
            # Tokenize
            ref_tokens_list = [ref.split() for ref in refs]
            hyp_tokens = hyp.split()
            
            # Compute n-grams for references
            refs_ngrams = []
            for ref_tokens in ref_tokens_list:
                ngrams_dict = {}
                for n in range(1, self.n + 1):
                    ngrams_dict.update(self._get_ngrams(ref_tokens, n))
                refs_ngrams.append(ngrams_dict)
            
            # Compute n-grams for hypothesis
            hyp_ngrams = {}
            for n in range(1, self.n + 1):
                hyp_ngrams.update(self._get_ngrams(hyp_tokens, n))
            
            all_refs_ngrams.append(refs_ngrams)
            all_hyps_ngrams.append(hyp_ngrams)
        
        # Compute document frequencies
        all_ngrams = []
        for refs_ngrams in all_refs_ngrams:
            for ngrams_dict in refs_ngrams:
                all_ngrams.append(ngrams_dict)
        
        doc_freq = self._compute_doc_freq(all_ngrams)
        num_docs = len(all_ngrams)
        
        # Compute CIDEr for each hypothesis
        for refs_ngrams, hyp_ngrams in zip(all_refs_ngrams, all_hyps_ngrams):
            # Compute TF-IDF vectors
            vec_hyp = {}
            vec_refs = []
            
            # Hypothesis vector
            for ngram, count in hyp_ngrams.items():
                tf = count / len(hyp_ngrams)
                idf = np.log((num_docs + 1) / (doc_freq[ngram] + 1))
                vec_hyp[ngram] = tf * idf
            
            # Reference vectors (average)
            for ref_ngrams in refs_ngrams:
                vec_ref = {}
                for ngram, count in ref_ngrams.items():
                    tf = count / len(ref_ngrams)
                    idf = np.log((num_docs + 1) / (doc_freq[ngram] + 1))
                    vec_ref[ngram] = tf * idf
                vec_refs.append(vec_ref)
            
            # Compute cosine similarity
            similarities = []
            for vec_ref in vec_refs:
                # Dot product
                dot_product = sum(vec_hyp.get(k, 0) * v for k, v in vec_ref.items())
                
                # Norms
                norm_hyp = np.sqrt(sum(v**2 for v in vec_hyp.values()))
                norm_ref = np.sqrt(sum(v**2 for v in vec_ref.values()))
                
                if norm_hyp > 0 and norm_ref > 0:
                    sim = dot_product / (norm_hyp * norm_ref)
                else:
                    sim = 0.0
                
                similarities.append(sim)
            
            # Average similarity
            score = np.mean(similarities) * 10.0  # Scale to 0-10
            scores.append(score)
        
        return np.mean(scores)


# In[151]:


class CaptionEvaluator:
    """
    Complete evaluator for image captioning
    Computes all major metrics
    """
    
    def __init__(self):
        self.bleu_scorer = BLEUScorer()
        self.meteor_scorer = METEORScorer()
        self.rouge_scorer = ROUGEScorer()
        self.cider_scorer = CIDErScorer()
    
    def evaluate(self, references: List[List[str]], hypotheses: List[str], 
                 verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate generated captions against references
        
        Args:
            references: List of [list of reference captions] for each image
            hypotheses: List of generated captions (one per image)
            verbose: Print results
        
        Returns:
            Dictionary with all metric scores
        """
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must match")
        
        results = {}
        
        # 1. BLEU scores
        if verbose:
            print("Computing BLEU scores...")
        bleu_scores = self.bleu_scorer.compute_corpus_bleu(references, hypotheses)
        results.update(bleu_scores)
        
        # 2. METEOR score
        if verbose:
            print("Computing METEOR score...")
        meteor = self.meteor_scorer.compute_corpus_meteor(references, hypotheses)
        results['METEOR'] = meteor
        
        # 3. ROUGE-L score
        if verbose:
            print("Computing ROUGE-L scores...")
        rouge_scores = self.rouge_scorer.compute_corpus_rouge(references, hypotheses)
        results.update(rouge_scores)
        
        # 4. CIDEr score
        if verbose:
            print("Computing CIDEr score...")
        cider = self.cider_scorer.compute_cider(references, hypotheses)
        results['CIDEr'] = cider
        
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
            print(f"METEOR:   {results['METEOR']:.4f}")
            print()
            print("ROUGE-L Scores:")
            print(f"  Precision: {results['ROUGE-L-P']:.4f}")
            print(f"  Recall:    {results['ROUGE-L-R']:.4f}")
            print(f"  F1-Score:  {results['ROUGE-L-F1']:.4f}")
            print()
            print(f"CIDEr:    {results['CIDEr']:.4f}")
            print("="*60)
        
        return results
    
    def evaluate_single(self, references: List[str], hypothesis: str) -> Dict[str, float]:
        """Evaluate single caption"""
        results = {}
        
        # BLEU
        for n in [1, 2, 3, 4]:
            score = self.bleu_scorer.compute_sentence_bleu(references, hypothesis, n)
            results[f'BLEU-{n}'] = score
        
        # METEOR
        results['METEOR'] = self.meteor_scorer.compute_sentence_meteor(references, hypothesis)
        
        # ROUGE-L
        rouge = self.rouge_scorer.compute_sentence_rouge(references, hypothesis)
        results['ROUGE-L-F1'] = rouge['f1']
        
        return results


# In[152]:


def evaluate_model(model, test_images: List[str], caption_dict: Dict[str, List[str]], 
                   img_folder: str, tokenizer, config, 
                   use_beam_search: bool = False) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        model: Trained captioning model
        test_images: List of test image names
        caption_dict: Dictionary mapping image names to reference captions
        img_folder: Folder containing images
        tokenizer: Fitted tokenizer
        config: Configuration object
        use_beam_search: Use beam search for generation
    
    Returns:
        Dictionary with all evaluation metrics
    """
    from image_captioning import generate_caption, generate_beam_search
    
    print(f"Evaluating on {len(test_images)} test images...")
    
    references = []
    hypotheses = []
    
    for i, img_name in enumerate(test_images):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(test_images)}")
        
        img_path = os.path.join(img_folder, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        # Get reference captions
        refs = caption_dict[img_name]
        references.append(refs)
        
        # Generate caption
        try:
            if use_beam_search:
                hyp = generate_beam_search(model, img_path, tokenizer, config)
            else:
                hyp = generate_caption(model, img_path, tokenizer, config)
            hypotheses.append(hyp)
        except Exception as e:
            print(f"Error generating caption for {img_name}: {e}")
            hypotheses.append("")  # Empty caption for failed generation
    
    # Evaluate
    evaluator = CaptionEvaluator()
    results = evaluator.evaluate(references, hypotheses, verbose=True)
    
    return results, references, hypotheses


# In[153]:


def analyze_predictions(references: List[List[str]], hypotheses: List[str], 
                        image_names: List[str] = None, n_samples: int = 10):
    """
    Show qualitative examples of predictions
    
    Args:
        references: Reference captions
        hypotheses: Generated captions
        image_names: Optional image names
        n_samples: Number of samples to show
    """
    evaluator = CaptionEvaluator()
    
    print("\n" + "="*80)
    print("QUALITATIVE ANALYSIS")
    print("="*80)
    
    # Select random samples
    indices = np.random.choice(len(hypotheses), min(n_samples, len(hypotheses)), replace=False)
    
    for idx in indices:
        refs = references[idx]
        hyp = hypotheses[idx]
        
        # Compute metrics for this sample
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


# In[154]:


def save_evaluation_results(results: Dict[str, float], references: List[List[str]], 
                            hypotheses: List[str], image_names: List[str] = None,
                            output_dir: str = "evaluation_results"):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save metrics
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 2. Save predictions
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
    
    # 3. Save as CSV for easy viewing
    df = pd.DataFrame({
        'image': image_names if image_names else range(len(hypotheses)),
        'generated': hypotheses,
        'reference_1': [refs[0] if len(refs) > 0 else "" for refs in references],
        'reference_2': [refs[1] if len(refs) > 1 else "" for refs in references],
    })
    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    print(f"\nResults saved to {output_dir}/")


# In[155]:


def example_usage():
    """Example of how to use the evaluation functions"""
    
    # Example data
    references = [
        ["a dog playing in the park", "a brown dog running on grass"],
        ["a cat sitting on a chair", "an orange cat on furniture"],
        ["a car on the street", "a red vehicle on the road"]
    ]
    
    hypotheses = [
        "a dog running in a park",
        "a cat sitting on a chair",
        "a red car on the street"
    ]
    
    # Evaluate
    evaluator = CaptionEvaluator()
    results = evaluator.evaluate(references, hypotheses, verbose=True)
    
    # Show qualitative analysis
    analyze_predictions(references, hypotheses, n_samples=3)
    
    return results


# In[156]:


def main_evaluation(model_path: str, test_images: List[str], 
                    caption_dict: Dict[str, List[str]], img_folder: str,
                    tokenizer_path: str, config, use_beam_search: bool = True):
    """
    Complete evaluation pipeline
    
    Args:
        model_path: Path to saved model
        test_images: List of test image filenames
        caption_dict: Dictionary of captions
        img_folder: Folder containing images
        tokenizer_path: Path to saved tokenizer
        config: Configuration object
        use_beam_search: Use beam search for generation
    """
    import tensorflow as tf
    import pickle
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = tf.keras.models.load_model(model_path)
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Evaluate
    results, references, hypotheses = evaluate_model(
        model, test_images, caption_dict, img_folder, 
        tokenizer, config, use_beam_search
    )
    
    # Qualitative analysis
    analyze_predictions(references, hypotheses, test_images, n_samples=10)
    
    # Save results
    save_evaluation_results(results, references, hypotheses, test_images)
    
    return results


# In[157]:


if __name__ == "__main__":
    # Simple test
    print("Testing evaluation metrics...\n")
    example_usage()

