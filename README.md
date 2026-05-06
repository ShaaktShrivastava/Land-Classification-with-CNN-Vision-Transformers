# Land Classification with CNN & Vision Transformers

End-to-end deep learning project for classifying satellite images into **agricultural** and **non-agricultural** land. Implements and compares CNN and CNN-ViT hybrid architectures in both **Keras/TensorFlow** and **PyTorch**.

---

## Project Structure

```
├── 01_Memory_vs_Generator_Data_Loading.ipynb     # Memory-based vs generator-based loading
├── 02_Keras_Data_Loading_and_Augmentation.ipynb  # Keras data pipeline & augmentation
├── 03_PyTorch_Data_Loading_and_Augmentation.ipynb # PyTorch data pipeline & augmentation
├── 04_Keras_CNN_Classifier.ipynb                 # Train & evaluate a Keras CNN
├── 05_PyTorch_CNN_Classifier.ipynb               # Train & evaluate a PyTorch CNN
├── 06_Model_Comparison_Keras_vs_PyTorch.ipynb    # Compare CNN models across frameworks
├── 07_Keras_Vision_Transformer.ipynb             # CNN-ViT hybrid model in Keras
├── 08_PyTorch_Vision_Transformer.ipynb           # CNN-ViT hybrid model in PyTorch
└── 09_CNN_ViT_Final_Evaluation.ipynb             # Final evaluation & metrics comparison
```

---

## Dataset

Satellite image tiles (64×64 RGB) split into two classes:

| Class | Label | Description |
|---|---|---|
| `class_0_non_agri` | 0 | Non-agricultural land |
| `class_1_agri` | 1 | Agricultural / cultivable land |

~6,000 images total. Downloaded automatically when running any notebook. Not included in this repo.

---

## Models

### CNN Baseline
- 6 convolutional blocks: `Conv2D → ReLU → MaxPool2D → BatchNorm`
- `GlobalAveragePooling → Dense classifier`
- Trained with `Adam (lr=0.001)`, `batch_size=128`
- Data augmentation: random rotation, flips, shear, zoom

### CNN-ViT Hybrid
- CNN backbone (above) for local feature extraction
- Feature map reshaped into tokens → positional embeddings added
- Vision Transformer encoder: multi-head self-attention + MLP blocks
- Classification via CLS token
- Keras: 4 transformer layers, 8 heads, embed_dim=768
- PyTorch: 3 transformer layers, 6 heads, embed_dim=768 (baseline) / 12 layers, 12 heads (deep variant)

---

## Results

### CNN Models (pre-trained, 20 epochs)

| Metric | Keras CNN | PyTorch CNN |
|---|---|---|
| Accuracy | 0.9925 | 0.9988 |
| Precision | 1.0000 | 0.9983 |
| Recall | 0.9850 | 0.9993 |
| F1 Score | 0.9924 | 0.9988 |
| ROC-AUC | 1.0000 | 1.0000 |

> PyTorch CNN leads on accuracy, recall, and F1. Keras CNN achieves perfect precision — zero false positives.
> Both reach perfect ROC-AUC, indicating flawless class separability.

### CNN-ViT Hybrid Models (pre-trained, 20 epochs)

| Metric | Keras CNN-ViT | PyTorch CNN-ViT |
|---|---|---|
| Accuracy | ~0.98 | ~0.99 |
| ROC-AUC | ~1.00 | ~1.00 |

> Exact values vary by run. See `09_CNN_ViT_Final_Evaluation.ipynb` for full metrics.


---

## Key Observations

- Both frameworks achieve >99% accuracy on this binary classification task
- PyTorch CNN has stronger recall — better at catching all agricultural land
- Keras CNN has perfect precision — never misclassifies non-agricultural land as agricultural
- CNN-ViT hybrids match or exceed pure CNN performance by combining local (CNN) and global (ViT) feature learning
- Training a deeper ViT (depth=12, heads=12) takes significantly longer per epoch with marginal accuracy gain on this dataset size

---

## Setup

```bash
pip install numpy==1.26.4 matplotlib==3.9.2 Pillow
pip install tensorflow==2.17.0
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn==1.5.2
```

Open any notebook in Jupyter and run all cells. The dataset and pre-trained model weights are downloaded automatically inside each notebook.

---

## Notes

- Model weights (`.keras`, `.pth`) and the dataset are excluded from this repo via `.gitignore` due to file size (up to 1.8 GB)
- Pre-trained weights are downloaded from IBM Cloud Object Storage inside each notebook
- Built as part of the IBM AI Capstone Project with Deep Learning course on Coursera
