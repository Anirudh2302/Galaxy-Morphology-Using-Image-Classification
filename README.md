🌌 Galaxy Morphology Classification using Deep Learning


📌 Overview

This project explores the use of deep learning techniques for galaxy morphology classification, specifically distinguishing between elliptical and spiral galaxies using image data from the Galaxy Zoo dataset.

The aim was to investigate how different model architectures perform on this task and to understand their behaviour using interpretability techniques. Alongside performance, I also focused on model robustness and reliability, which are important in real-world scientific applications.

🎯 Research Question

How effectively can deep learning models classify galaxy morphology, and which architecture provides the best performance and robustness?


🎯 Objectives

Build and evaluate multiple deep learning models for image classification

Compare model performance using appropriate evaluation metrics

Analyse robustness by introducing noise into the dataset

Apply interpretability techniques (Grad-CAM) to understand model decisions

📊 Dataset

Source: Galaxy Zoo 2

Task: Binary classification (Elliptical vs Spiral)

Preprocessing steps:

Image resizing (224 × 224)

Normalization

Train / Validation / Test split

Efficient data pipeline using TensorFlow

🧠 Models Implemented
Custom CNN – baseline convolutional architecture
Tuned CNN – improved using hyperparameter tuning
EfficientNetB0 – transfer learning-based model
DeiT Transformer – vision transformer architecture
⚙️ Methodology
Models were trained using early stopping to prevent overfitting
Hyperparameters were tuned to improve CNN performance
Evaluation was done using accuracy, ROC-AUC, and confusion matrix
Noise experiments were conducted to test robustness
Grad-CAM was used to visualise model attention
📈 Results
Model	Accuracy	AUC	Errors
Custom CNN	98.36%	0.9989	132
Tuned CNN	97.58%	0.9974	196
EfficientNetB0	97.59%	0.9974	195
DeiT Transformer	98.83%	0.9994	94
🧪 Noise Experiment (Robustness)
Model	Accuracy	AUC	Errors
EfficientNet	90.33%	0.9671	1981
DeiT	92.11%	0.9781	1615

👉 The DeiT model demonstrated better robustness under noisy conditions, indicating stronger generalisation.

🔍 Interpretability (Grad-CAM)

Grad-CAM was applied to visualise which regions influenced model predictions.
The results showed that models consistently focused on central galaxy structures, which aligns with domain knowledge and increases trust in the model.

📌 Key Findings
Transformer-based models (DeiT) achieved the best performance
CNN-based models still performed strongly with proper tuning
Noise significantly impacted performance, but DeiT remained more stable
Grad-CAM confirmed that models learned meaningful features
🧾 Conclusion

This project demonstrates that deep learning, particularly vision transformers, is highly effective for galaxy classification tasks.

Beyond accuracy, incorporating robustness testing and interpretability provides a more complete evaluation, which is essential for applying machine learning in scientific domains.

🚀 Future Work
Use larger and more diverse astronomical datasets
Explore advanced transformer architectures
Improve robustness to noise and real-world distortions
Deploy as an automated classification tool
