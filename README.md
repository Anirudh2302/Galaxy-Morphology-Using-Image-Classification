# 🌌 Galaxy Morphology Classification using Deep Learning

## 📌 Abstract  
This project focuses on classifying galaxies into **elliptical and spiral types** using deep learning models trained on the Galaxy Zoo dataset.  
Multiple architectures including CNNs, transfer learning models, and transformers were evaluated, along with robustness and interpretability analysis.

---

## ❓ Research Question  
How effectively can deep learning models classify galaxy morphology, and which model performs best?

---

## 🎯 Objectives  
- Train multiple deep learning models  
- Compare performance using accuracy and AUC  
- Test robustness using noisy data  
- Apply Grad-CAM for interpretability  

---

## 📊 Dataset  
- Galaxy Zoo 2 dataset  
- Binary classification (Elliptical vs Spiral)  
- Images resized to 224 × 224  
- Train / Validation / Test split  

---

## 🧠 Models Used  
- Custom CNN  
- Tuned CNN  
- EfficientNetB0  
- DeiT Transformer  

---

## ⚙️ Methodology  
- Training with early stopping  
- Hyperparameter tuning for CNN  
- Evaluation using Accuracy, AUC, Confusion Matrix  
- Noise-based robustness testing  
- Grad-CAM for visualization  

---

## 📈 Results  

| Model            | Accuracy | AUC     | Errors |
|------------------|----------|---------|--------|
| Custom CNN       | 98.36%   | 0.9989  | 132    |
| Tuned CNN        | 97.58%   | 0.9974  | 196    |
| EfficientNetB0   | 97.59%   | 0.9974  | 195    |
| DeiT Transformer | **98.83%** | **0.9994** | **94** |

---

## 🧪 Noise Experiment  

| Model        | Accuracy | AUC     | Errors |
|--------------|----------|---------|--------|
| EfficientNet | 90.33%   | 0.9671  | 1981   |
| DeiT         | 92.11%   | 0.9781  | 1615   |

---

## 🔍 Interpretability (Grad-CAM)  
Grad-CAM was used to visualize model focus areas.  
Models consistently highlighted meaningful galaxy structures, improving trust in predictions.


## 📌 Key Findings  
- DeiT Transformer achieved the best performance  
- CNN models performed strongly as baselines  
- Performance drops under noise, but DeiT remains more robust  
- Grad-CAM confirms meaningful feature learning  

---

## 🧾 Conclusion  
Deep learning models, especially transformer-based architectures, are highly effective for galaxy classification tasks.  
This project demonstrates strong performance, robustness, and interpretability.

---

## 🚀 Future Work  
- Multi-class classification  
- Larger datasets  
- Better noise handling  
- Deployment as real-world system  



  
