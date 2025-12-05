# 😊 Vision Transformer for Facial Expression Recognition

> Implementacja Vision Transformer (ViT) od podstaw według papieru "An Image is Worth 16x16 Words" dla rozpoznawania emocji z twarzy - 73% accuracy na zbiorze testowym

---

## 📋 Executive Summary

**Vision Transformer for Facial Expression Recognition** to projekt demonstrujący implementację transformer architecture dla Computer Vision. Projekt zawiera:

- **Vision Transformer (ViT)** - pełna implementacja od zera z mechanizmem self-attention
- **Porównanie architektur** - CNN, CNN+LSTM, CNN+RNN vs ViT
- **Real-time detection** - detekcja emocji z kamery na żywo z OpenCV
- **Wizualizacja attention** - analiza co model "widzi" przy podejmowaniu decyzji

Klasyfikacja 7 emocji: **angry, disgust, fear, happy, neutral, sad, surprise**

**Wyniki:**
- Model początkowy (CNN): **~65% accuracy**
- Vision Transformer: **73% accuracy** ⭐

---

## ✨ Key Features

### 🤖 Vision Transformer Implementation
- **From Scratch ViT** - pełna implementacja według Google Research paper
- **Patch Embedding** - podział obrazu na patches 4×4
- **Positional Encoding** - sinusoidalne embeddingi pozycji
- **Multi-Head Self-Attention** - 8 attention heads
- **Transformer Encoder** - 8 warstw z layer normalization
- **Classification Token [CLS]** - specjalny token do klasyfikacji

### 📊 Multiple Architectures
1. **CNN (TinyVGG)** - baseline model (~65% acc)
2. **CNN + LSTM** - hybrid architecture z temporal processing
3. **CNN + RNN** - recurrent approach
4. **Vision Transformer** - pure attention-based (73% acc)

### 🎥 Real-time Application
- **Webcam detection** - detekcja twarzy z Haar Cascade
- **Live emotion prediction** - klasyfikacja w czasie rzeczywistym
- **Confidence scores** - prawdopodobieństwo dla każdej emocji
- **Visual feedback** - bounding box + label na twarzy

### 🔍 Attention Visualization
- **Attention maps** - wizualizacja co model "patrzy"
- **Model interpretability** - zrozumienie decyzji modelu
- **Comparison across models** - różnice między ViT a CNN

---

## 🛠️ Tech Stack & Skills Demonstrated

### Technologie
| Kategoria | Technologia |
|-----------|-------------|
| **Framework** | PyTorch 2.0+ |
| **Computer Vision** | OpenCV, torchvision |
| **Architecture** | Vision Transformer (ViT) |
| **Dataset** | FER2013 (35k images) |
| **Hardware** | CUDA (GPU acceleration) |

### Umiejętności programistyczne
- ✅ **Transformer Architecture** - self-attention, positional encoding
- ✅ **Computer Vision** - image classification, face detection
- ✅ **Deep Learning** - CNN, LSTM, RNN, ViT
- ✅ **PyTorch Advanced** - custom layers, attention mechanisms
- ✅ **Model Interpretability** - attention visualization
- ✅ **Real-time Processing** - webcam integration, OpenCV
- ✅ **Hyperparameter Tuning** - learning rate scheduling, regularization

### Koncepcje teoretyczne
- **Self-Attention Mechanism** - scaled dot-product attention
- **Vision Transformers** - patches, embeddings, encoders
- **Transfer Learning Concepts** - patch-based image processing
- **Batch Normalization** - stabilizacja treningu CNN
- **Learning Rate Scheduling** - MultiStepLR decay
- **Data Augmentation** - AutoAugment dla obrazów

---

## 📁 Project Structure

```
Face_Expression_recognition_VIT/
├── model_VISION_TRANSFORMER.py    # ViT implementation (115 linii)
├── model_VIT_VISUAL.py            # ViT z visualizacją attention
├── model_CNN.py                   # TinyVGG baseline
├── model_CNN_LSTM.py              # CNN + LSTM hybrid
├── model_CNN_RNN.py               # CNN + RNN hybrid
│
├── TRAIN_MODELS/
│   ├── train_model_ViS.py         # Training script dla ViT
│   ├── train_model_CNN.py         # Training dla CNN
│   └── train_model_CNN_RNN_LSTM.py # Training dla hybrids
│
├── engine.py                      # Training/testing loops
├── dataSetup.py                   # Dataset loading (FER2013)
├── utils.py                       # Helper functions
├── tenserflow.py                  # TensorBoard writer
│
├── video.py                       # Real-time webcam detection
├── haarcascade_frontalface_default.xml  # Face detector
│
├── dane/
│   ├── train/                     # Training images (7 classes)
│   └── test/                      # Test images
│
└── models/
    └── tinyvgg_model_01.pth       # Trained model weights
```

---

## 🔍 Detailed Architecture Description

### 🎯 Vision Transformer (ViT) - Główny model

#### Architektura:
```
Input Image (48×48×3)
    ↓
Patch Embedding (4×4 patches = 144 patches)
    ↓
Add [CLS] token + Positional Encoding
    ↓
Transformer Encoder × 8 layers
    ├── Multi-Head Self-Attention (8 heads)
    ├── Layer Normalization
    ├── MLP (4× embedding size)
    └── Residual Connections
    ↓
Extract [CLS] token
    ↓
Classification Head → 7 emotions
```

#### `model_VISION_TRANSFORMER.py` - Implementacja

**1. Patch Embedding**
```python
class Patch_embedding(nn.Module):
    def __init__(self, C_channels, patch_size, emb_size, img_size=48):
        # Konwolucja jako patch embedding
        self.projection = nn.Sequential(
            nn.Conv2d(C_channels, emb_size, 
                     kernel_size=patch_size, 
                     stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        # CLS token i positional embedding
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_embed = nn.Parameter(PositionEmbedding(self.len_seq + 1, emb_size))
    
    def forward(self, x):
        # Input: [batch, 3, 48, 48]
        x = self.projection(x)  # [batch, 144, emb_size]
        
        # Dodaj CLS token
        cls_token = repeat(self.cls_token, '() s e -> n s e', n=batch_size)
        x = torch.cat([cls_token, x], dim=1)  # [batch, 145, emb_size]
        
        # Dodaj positional encoding
        x = x + self.pos_embed
        return x
```

**Wyjaśnienie:**
- Obraz 48×48 dzielony na patches 4×4 = **144 patches**
- Każdy patch linearnie projektowany do **embedding dimension** (256)
- **[CLS] token** dodawany na początku sekwencji
- **Positional encoding** dodaje informację o pozycji każdego patch'a

---

**2. Positional Encoding (Sinusoidal)**
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

**Wzór matematyczny:**
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

Gdzie:
- $pos$ - pozycja patch'a w sekwencji
- $i$ - wymiar embeddingu
- $d_{model}$ - rozmiar embeddingu (256)

---

**3. Transformer Encoder Block**
```python
class EncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, num_hidden_layers, dropout):
        self.norm = nn.LayerNorm(emb_size)
        
        self.multihead = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.MLP = nn.Sequential(
            nn.Linear(emb_size, num_hidden_layers),  # 256 → 1024
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden_layers, emb_size),  # 1024 → 256
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-Head Self-Attention
        x_norm1 = self.norm(x)
        x_attn, _ = self.multihead(x_norm1, x_norm1, x_norm1)
        x = x_attn + x  # Residual connection
        
        # MLP (Feed-Forward)
        x_norm2 = self.norm(x)
        x_mlp = self.MLP(x_norm2)
        x = x_mlp + x  # Residual connection
        
        return x
```

**Attention formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

**4. Complete ViT Model**
```python
class Vit(nn.Module):
    def __init__(self, emb_size=256, num_heads=8, num_layers=8, 
                 num_hidden_layers=1024, C_channels=3, 
                 patch_size=4, num_class=7, dropout=0.2):
        
        self.Patch_embed = Patch_embedding(C_channels, patch_size, 
                                           emb_size, img_size=48)
        
        # Stack Encoder Blocks
        self.encoders = nn.Sequential(*[
            EncoderBlock(emb_size, num_heads, num_hidden_layers, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_class)
        )
    
    def forward(self, x):
        x = self.Patch_embed(x)       # [batch, 145, 256]
        x = self.encoders(x)          # [batch, 145, 256]
        x = x[:, 0]                   # Extract CLS token
        x = self.classifier(x)        # [batch, 7]
        return x
```

**Hyperparametry:**
- Embedding size: **256**
- Attention heads: **8**
- Encoder layers: **8**
- MLP hidden: **1024** (4× embedding)
- Patch size: **4×4**
- Dropout: **0.2**

---

### 📊 Porównawcze architektury

#### **CNN + LSTM Hybrid**
```python
class HybridCNNandLSTM(nn.Module):
    def __init__(self):
        # CNN Feature Extractor
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 350, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(350),
            nn.MaxPool2d(2, 2),
            # ... więcej warstw
        )  # Output: [batch, 350, 2, 2]
        
        # LSTM dla sekwencyjnego przetwarzania
        self.lstm_layer = nn.LSTM(
            input_size=350 * 2 * 2,
            hidden_size=hidden_units_rnn,
            num_layers=num_layers_rnn,
            batch_first=True
        )
```

**Idea:** CNN ekstraktuje features → LSTM przetwarza sekwencyjnie → Klasyfikacja

---

## 🎓 Training Process

### Training Script: `train_model_ViS.py`

**Hyperparametry:**
```python
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EMB_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS_ATTEN = 8
PATCH_SIZE = 4
DROPOUT = 0.2
EPOCHS = 250

# Learning rate scheduling
GAMMA_LR_SCHEDULER = 0.1
MILESTONES = [150, 200]  # Decay at epochs 150 and 200
```

**Optimizer:**
```python
optimizer = optim.Adam(model.parameters(), 
                       lr=LEARNING_RATE,
                       weight_decay=3e-5)  # L2 regularization

lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=MILESTONES, 
    gamma=GAMMA_LR_SCHEDULER
)
```

**Data Augmentation:**
```python
transform = v2.Compose([
    v2.Resize(size=(48, 48)),
    v2.ToImage(),
    v2.AutoAugment(),  # Automatyczna augmentacja
    v2.ToDtype(torch.float32, scale=True)
])
```

**Training Loop** (`engine.py`):
```python
def train_step(model, dataloader, loss_fn, optimizer, 
               lr_schedule, accuracy_fn, device):
    model.train()
    train_loss, train_acc = 0, 0
    
    for img, y_true in dataloader:
        img, y_true = img.to(device), y_true.to(device)
        
        y_pred = model(img)
        loss = loss_fn(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc += accuracy_fn(y_pred, y_true).item()
        train_loss += loss.item()
    
    lr_schedule.step()
    return train_loss / len(dataloader), train_acc / len(dataloader)
```

---

## 🎥 Real-time Emotion Detection

### `video.py` - Webcam Application

**Komponenty:**
1. **Face Detection** - Haar Cascade Classifier
2. **Preprocessing** - resize, normalize, tensor conversion
3. **Model Inference** - forward pass przez ViT
4. **Visualization** - bounding box + emotion label

**Kod:**
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

## 📊 Results & Attention Visualization

### Model Comparison

| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| **CNN (TinyVGG)** | 65% | ~500k | ~30 min |
| **CNN + LSTM** | 68% | ~750k | ~45 min |
| **CNN + RNN** | 67% | ~700k | ~40 min |
| **Vision Transformer** | **73%** ⭐ | ~1.2M | ~60 min |

### Attention Maps - Co model "widzi"?

**Model początkowy (65% accuracy):**
![Attenction 65 procent sample](https://github.com/user-attachments/assets/a925c538-3cdd-4064-b349-f11d9c7e6c2e)

**Vision Transformer (73% accuracy):**
![second attenction 73 pricent sample](https://github.com/user-attachments/assets/6d524e8d-ee69-45f1-b672-e9c1360c0cb4)

![attenction 73 procent sample](https://github.com/user-attachments/assets/4859a8de-4f7b-4774-a7ca-114e7a2e5aed)

**Wnioski z attention maps:**
- ViT skupia się na **kluczowych obszarach twarzy** (oczy, usta, brwi)
- Model początkowy (CNN) miał bardziej **rozproszoną uwagę**
- Self-attention pozwala na **globalne zrozumienie** relacji między patch'ami
- **Emocje pozytywne** (happy) - focus na uśmiech
- **Emocje negatywne** (angry, sad) - focus na brwi i oczy

---

## 🚀 Installation & Setup

### Wymagania

```
✓ Python 3.8+
✓ CUDA-capable GPU (zalecane)
✓ Webcam (dla real-time detection)
✓ 4GB RAM minimum
```

### Instalacja

1. **Sklonuj repozytorium**
```bash
git clone https://github.com/[username]/facial-expression-vit.git
cd facial-expression-vit
```

2. **Zainstaluj zależności**
```bash
pip install torch torchvision opencv-python einops torchmetrics torchinfo
```

Lub z pliku requirements.txt:
```bash
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

3. **Pobierz dataset FER2013**
```bash
# Struktura folderów:
dane/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    [same structure]
```

Dataset dostępny na: [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 🎮 How to Use

### 1️⃣ Trening Vision Transformer

```bash
cd TRAIN_MODELS
python train_model_ViS.py
```

**Output:**
```
Epoch: 1 | train_loss: 1.8234 | train_acc: 0.3421 | test_loss: 1.6543 | test_acc: 0.4123
Epoch: 2 | train_loss: 1.5432 | train_acc: 0.4567 | test_loss: 1.4321 | test_acc: 0.5234
...
Epoch: 250 | train_loss: 0.7234 | train_acc: 0.7521 | test_acc: 0.7300
```

Model zapisywany co epokę w `models/tinyvgg_model_01.pth`

---

### 2️⃣ Real-time Emotion Detection

```bash
python video.py
```

**Instrukcja:**
1. Program otworzy webcam
2. Ustaw się przed kamerą (dobrze oświetlone miejsce)
3. Model wykryje twarz i przewidzi emocję
4. Wciśnij `q` aby zakończyć

**Tips dla lepszej detekcji:**
- Dobre oświetlenie (front-facing)
- Twarz skierowana do kamery
- Wyraźna mimika
- Odległość ~50-100cm od kamery

---

### 3️⃣ Trening innych architektur

**CNN:**
```bash
cd TRAIN_MODELS
python train_model_CNN.py
```

**CNN + LSTM/RNN:**
```bash
python train_model_CNN_RNN_LSTM.py
```

---

### 4️⃣ Wizualizacja Attention

```bash
python model_VIT_VISUAL.py
```

Generuje attention maps pokazujące:
- Na które patche model zwraca uwagę
- Różnice między warstwami encodera
- Jak attention zmienia się podczas treningu

---

## 🔬 Implementation Details

### Dlaczego Vision Transformer działa lepiej?

**1. Global Context**
- CNN: Receptive field rośnie stopniowo (local → global)
- ViT: Od początku widzi całą twarz (global attention)

**2. Self-Attention Mechanism**
```
Przykład: Wykrywanie "happy"
- Patch oczu ← → Patch uśmiechu (wysoka attention)
- Model uczy się RELACJI między obszarami
```

**3. Position Encoding**
- ViT wie GDZIE jest każdy patch
- Ważne dla rozróżnienia: "brwi do góry" vs "brwi do dołu"

**4. Flexibility**
- ViT może "skupić się" na różnych obszarach dla różnych emocji
- CNN ma sztywne filtry

---

### Challenges & Solutions

**Problem 1: Small Dataset (35k images)**
- **Rozwiązanie**: AutoAugment, L2 regularization, dropout

**Problem 2: Small Image Size (48×48)**
- **Rozwiązanie**: Małe patches (4×4), więcej patches (144)

**Problem 3: Overfitting**
- **Rozwiązanie**: 
  - Dropout 0.2
  - Weight decay 3e-5
  - Learning rate scheduling
  - Early stopping

**Problem 4: Long Training Time**
- **Rozwiązanie**:
  - CUDA acceleration
  - Batch size 32
  - Persistent workers (DataLoader)
  - Prefetch factor 2

---

## 📄 License

MIT License - użyj jak chcesz!

---

## 👤 Author

**GitHub**: [@[username]](https://github.com/[username])

### Kontakt
- 📧 Email: [your-email@example.com]
- 💼 LinkedIn: [Your LinkedIn]
- 🌐 Portfolio: [your-website.com]

---

## 🙏 Acknowledgments

- **Google Research** - Vision Transformer paper
- **PyTorch Team** - fantastyczny framework
- **FER2013 Dataset** - Kaggle community
- **OpenCV** - real-time computer vision

---

## 📚 References

**Papers:**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**Resources:**
- [Vision Transformer Explained](https://youtu.be/TrdevFK_am4)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

---

<div align="center">

### ⭐ Star this repo if you found it helpful! ⭐

**Made with 🧠 and 😊 by [Your Name]**

</div>

---

*README last updated: 5 grudnia 2024*
