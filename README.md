# Vision Transformer Implementation

PyTorch implementation of Vision Transformer (ViT) from scratch based on "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". Tested on multiple datasets including FER2013 (facial expressions) and CIFAR-10 (objects).

**Note**: This is a universal ViT implementation. The attention visualizations show examples from different datasets - airplane images are from CIFAR-10 testing, while the main application focuses on FER2013 emotion recognition.

## About

Four architectures for image classification:
1. **CNN (TinyVGG)** - baseline (~65% FER2013)
2. **CNN + LSTM** - hybrid with temporal processing
3. **CNN + RNN** - recurrent approach  
4. **Vision Transformer** - pure attention-based (**73% FER2013**)

**Real-time emotion detection** from webcam with OpenCV Haar Cascade face detection.

## Features

- **ViT from scratch**: Patch embedding, positional encoding, multi-head self-attention, transformer encoders
- **7-class emotion classification**: angry, disgust, fear, happy, neutral, sad, surprise
- **Attention visualization**: Interpretability analysis showing model focus areas
- **Multiple datasets**: Tested on FER2013 (faces) and CIFAR-10 (objects)
- **Webcam integration**: Real-time face detection + emotion prediction

## Tech Stack

- **Framework**: PyTorch 2.0+
- **CV Libraries**: OpenCV, torchvision, einops
- **Dataset**: FER2013 (35k images, 48×48 grayscale)
- **Hardware**: CUDA GPU acceleration

## Project Structure

```
Face_Expression_recognition_VIT/
├── model_VISION_TRANSFORMER.py    # ViT implementation
├── model_VIT_VISUAL.py            # ViT with attention visualization
├── model_CNN.py                   # TinyVGG baseline
├── model_CNN_LSTM.py              # CNN + LSTM hybrid
├── model_CNN_RNN.py               # CNN + RNN hybrid
│
├── TRAIN_MODELS/
│   ├── train_model_ViS.py         # ViT training script
│   ├── train_model_CNN.py         # CNN training
│   └── train_model_CNN_RNN_LSTM.py # Hybrid training
│
├── engine.py                      # Train/test loops
├── dataSetup.py                   # FER2013 dataloader
├── utils.py                       # Helper functions
├── tensorboard.py                 # TensorBoard logging
│
├── video.py                       # Real-time webcam detection
├── haarcascade_frontalface_default.xml  # Face detector
│
├── dane/
│   ├── train/                     # 7 emotion classes
│   └── test/
│
└── models/
    └── tinyvgg_model_01.pth       # Trained weights
```

## Vision Transformer Architecture

```
Input (48×48×3) → Patch Embedding (4×4 patches = 144 patches) 
  → Add [CLS] + Positional Encoding 
  → Transformer Encoder × 8 layers (Multi-Head Attention + MLP + LayerNorm + Residuals)
  → Extract [CLS] token → FC(7 emotions)
```

### Implementation - `model_VISION_TRANSFORMER.py`

**1. Patch Embedding**
```python
class PatchEmbedding(nn.Module):
    def __init__(self, C_channels, patch_size, emb_size, img_size=48):
        self.projection = nn.Sequential(
            nn.Conv2d(C_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_embed = nn.Parameter(PositionEmbedding(self.len_seq + 1, emb_size))
    
    def forward(self, x):
        x = self.projection(x)  # [batch, 144, 256]
        cls_token = repeat(self.cls_token, '() s e -> n s e', n=batch_size)
        x = torch.cat([cls_token, x], dim=1)  # [batch, 145, 256]
        x = x + self.pos_embed
        return x
```

48×48 image → 4×4 patches → 144 patches → embedding dim 256 → add [CLS] token → add positional encoding

**2. Positional Encoding**
```python
def PositionEmbedding(seq_len, emb_size):
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_len):
        for j in range(emb_size):
            if j % 2 == 0:
                embeddings[i][j] = np.sin(i / (10000 ** (j / emb_size)))
            else:
                embeddings[i][j] = np.cos(i / (10000 ** ((j-1) / emb_size)))
    return embeddings
```

Formula: $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$

**3. Encoder Block + Complete Model**
```python
class Vit(nn.Module):
    def __init__(self, emb_size=256, num_heads=8, num_layers=8, 
                 num_hidden_layers=1024, C_channels=3, 
                 patch_size=4, num_class=7, dropout=0.2):
        
        self.Patch_embed = PatchEmbedding(C_channels, patch_size, emb_size, img_size=48)
        
        self.encoders = nn.Sequential(*[
            EncoderBlock(emb_size, num_heads, num_hidden_layers, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, num_class))
    
    def forward(self, x):
        x = self.Patch_embed(x)       # [batch, 145, 256]
        x = self.encoders(x)          # [batch, 145, 256]
        x = x[:, 0]                   # Extract [CLS] token
        x = self.classifier(x)        # [batch, 7]
        return x
```

**Hyperparameters**: emb_size=256, num_heads=8, num_layers=8, MLP_hidden=1024 (4×emb), patch_size=4×4, dropout=0.2

**Attention formula**: $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

## Training

**Hyperparameters** (`train_model_ViS.py`):
```python
BATCH_SIZE = 32, LEARNING_RATE = 0.0001, EPOCHS = 250
EMB_SIZE = 256, NUM_HEADS = 8, NUM_LAYERS = 8, PATCH_SIZE = 4, DROPOUT = 0.2
MILESTONES = [150, 200], GAMMA_LR_SCHEDULER = 0.1  # LR decay

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=3e-5)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)
transform = v2.Compose([v2.Resize((48, 48)), v2.AutoAugment(), v2.ToDtype(torch.float32, scale=True)])
```

## Real-time Webcam Detection

**`video.py` implementation:**
```python
cap = cv2.VideoCapture(0)  # Otwórz webcam
head_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    colored = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    heads = head_cascade.detectMultiScale(
        colored, 
        scaleFactor=1.05, 
        minNeighbors=5, 
        minSize=(40, 40)
    )
    
    for (x, y, w, h) in heads:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract and preprocess face
        face = frame[y:y+h, x:x+w]
        face_tensor = torch.from_numpy(face).permute(2, 1, 0)
        face_tensor = transform(face_tensor).unsqueeze(0).to(device)
        
        # Predict emotion
        with torch.no_grad():
            pred = model(face_tensor)
            probs = F.softmax(pred, dim=1)
            confidence = probs.max().item() * 100
            emotion = class_names[probs.argmax().item()]
        
        # Display result
        cv2.putText(frame, f"{emotion}: {confidence:.0f}%", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## Results

| Model | Accuracy | Params | Train Time |
|-------|----------|--------|------------|
| CNN (TinyVGG) | 65% | ~500k | ~30 min |
| CNN + LSTM | 68% | ~750k | ~45 min |
| CNN + RNN | 67% | ~700k | ~40 min |
| **Vision Transformer** | **73%** | ~1.2M | ~60 min |

### Attention Visualization

**Baseline CNN (65% accuracy):**
![Attention 65%](https://github.com/user-attachments/assets/a925c538-3cdd-4064-b349-f11d9c7e6c2e)

**Vision Transformer (73% accuracy):**
![Attention 73% #1](https://github.com/user-attachments/assets/6d524e8d-ee69-45f1-b672-e9c1360c0cb4)
![Attention 73% #2](https://github.com/user-attachments/assets/4859a8de-4f7b-4774-a7ca-114e7a2e5aed)

**Observations:**
- ViT focuses on key facial features (eyes, mouth, eyebrows)
- CNN has more diffuse attention
- Self-attention enables global understanding of patch relationships
- Happy emotions: focus on smile; Angry/sad: focus on eyebrows and eyes

## Installation

**Requirements:**
- Python 3.8+, CUDA GPU (recommended), Webcam (for real-time detection), 4GB RAM

```bash
git clone https://github.com/[username]/facial-expression-vit.git
cd facial-expression-vit
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
einops>=0.7.0
torchmetrics>=1.2.0
torchinfo>=1.8.0
numpy>=1.24.0
matplotlib>=3.7.0
```

**Dataset**: Download FER2013 from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013), organize as:
```
dane/
  train/[angry, disgust, fear, happy, neutral, sad, surprise]/
  test/[same structure]
```

## Usage

**Train ViT:**
```bash
cd TRAIN_MODELS
python train_model_ViS.py
```

**Real-time emotion detection:**
```bash
python video.py  # Press 'q' to quit
```

**Train other models:**
```bash
python train_model_CNN.py  # Baseline CNN
python train_model_CNN_RNN_LSTM.py  # Hybrid models
```

**Visualize attention:**
```bash
python model_VIT_VISUAL.py
```

## Why ViT Works Better

1. **Global Context**: ViT sees entire face from start vs CNN's gradual receptive field growth
2. **Self-Attention**: Learns relationships between patches (e.g., eyes ← → smile for "happy")
3. **Position Encoding**: Knows patch locations (distinguishes "eyebrows up" vs "eyebrows down")
4. **Flexibility**: Adaptive focus per emotion vs CNN's fixed filters

**Challenges Solved:**
- Small dataset (35k): AutoAugment + L2 reg + dropout
- Small images (48×48): Small patches (4×4) → more patches (144)
- Overfitting: Dropout 0.2, weight decay 3e-5, LR scheduling
- Training time: CUDA + batch size 32 + persistent workers

## References

- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

MIT License

## Author

**GitHub**: [@[username]](https://github.com/[username])
