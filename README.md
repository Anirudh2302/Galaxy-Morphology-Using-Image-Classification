# Galaxy Morphology Classification using Deep Learning

A comparative study of convolutional, transfer-learning, and transformer approaches for binary classification of galaxies (spiral vs elliptical) on the Galaxy Zoo 2 (GZ2) dataset, with additional experiments on label-noise robustness and Grad-CAM interpretability.

**Author:** Karthikeya Anirudh Saraswatula (SRN 24099418)
**Module:** 7PAM2002 — MSc Data Science Project, University of Hertfordshire
**Supervisor:** Hasan Al Madfai

---

## Headline results

All models were trained and evaluated on the same stratified 70/15/15 split of a class-balanced, high-confidence (GZ2 vote fraction ≥ 0.8) subset of 53,978 galaxies.

| Model                | Accuracy | ROC-AUC | Errors (of 8,097) |
|----------------------|:--------:|:-------:|:-----------------:|
| **DeiT-Tiny**        | 0.9884   | 0.9994  | 94                |
| **Custom CNN**       | 0.9837   | 0.9989  | 132               |
| EfficientNetB0       | 0.9759   | 0.9974  | 195               |
| Tuned CNN            | 0.9758   | 0.9974  | 196               |

Robustness to label noise (GZ2 threshold lowered from 0.8 → 0.5):

| Model           | Clean Acc | Noisy Acc | Clean AUC | Noisy AUC |
|-----------------|:---------:|:---------:|:---------:|:---------:|
| EfficientNetB0  | 0.9759    | 0.9033    | 0.9974    | 0.9672    |
| DeiT-Tiny       | 0.9884    | 0.9212    | 0.9994    | 0.9781    |

DeiT degrades less than EfficientNet on both metrics, and both pretrained models keep AUC > 0.96 under noisier labels.

See [`report/24099418_Final_report.docx`](24099418_Final_report.docx) for the full write-up and [`results/final_results_all_models.json`](final_results_all_models.json) for machine-readable metrics including per-class precision/recall/F1.

---

## Repository layout

```
Galaxy-Morphology-Using-Image-Classification/
├── README.md                              ← this file
├── LICENSE                                 ← MIT
├── requirements.txt                        ← pinned Python dependencies
├── .gitignore
│
├── galaxy_morphology_deep_learning.ipynb   ← the main notebook (all training + evaluation)
│
├── figures/                                ← every figure used in the report
│   ├── galaxy_examples.png
│   ├── high_threshold_distribution.png
│   ├── balanced_dataset_distribution.png
│   ├── custom_cnn_accuracy.png
│   ├── custom_cnn_loss.png
│   ├── custom_cnn_confusion_matrix.png
│   ├── model_accuracy_comparison.png
│   ├── model_auc_comparison.png
│   ├── noise_accuracy.png
│   ├── noise_auc.png
│   └── grad_cam_examples.png
│
├── results/
│   └── final_results_all_models.json       ← metrics, hyperparameters, per-class scores
│
└── report/
    └── 24099418_Final_report.docx          ← submitted MSc project report
```

---

## What the notebook does

The notebook (`galaxy_morphology_deep_learning.ipynb`) is organised as a linear pipeline:

1. **Mount Google Drive and load GZ2** — JPEG image archive + `zoo2MainSpecz.csv.gz` catalogue with debiased vote fractions.
2. **Build the binary label** from three debiased columns (`t01_smooth_or_features_a01_smooth_debiased`, `t01_smooth_or_features_a02_features_or_disk_debiased`, `t04_spiral_a08_spiral_debiased`) at a confidence threshold of 0.8.
3. **Balance and split** — undersample to the minority class, stratified 70/15/15 train/val/test with seed 42.
4. **Train four models on identical splits**:
   - Custom CNN (4 conv blocks, Adam, 128×128 input)
   - Tuned CNN (Keras Tuner random search, 8 trials × 10 epochs, best config retrained)
   - EfficientNetB0 with two-stage fine-tuning (stage 1: head only @ 1e-3; stage 2: top 30 layers unfrozen @ 1e-5)
   - DeiT-Tiny from `timm`, fine-tuned end-to-end with AdamW @ 2e-5
5. **Evaluate** on the shared held-out test split: accuracy, ROC-AUC, confusion matrix, per-class precision/recall/F1.
6. **Noise experiment** — rebuild the dataset at threshold 0.5, retrain EfficientNet and DeiT, compare clean vs noisy performance.
7. **Grad-CAM** on EfficientNet's final convolutional block for one correct elliptical, one correct spiral and one misclassified galaxy.

Fixed seed (42) is set wherever the framework exposes it. Minor non-determinism from GPU kernels is expected.

---

## Reproducing the results

1. **Clone** this repository.
   ```bash
   git clone https://github.com/Anirudh2302/Galaxy-Morphology-Using-Image-Classification.git
   cd Galaxy-Morphology-Using-Image-Classification
   ```

2. **Install dependencies** (ideally in a virtual environment with a CUDA-enabled GPU):
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Galaxy Zoo 2 data** from <https://data.galaxyzoo.org>:
   - The GZ2 JPEG image archive (~243,500 images)
   - `zoo2MainSpecz.csv.gz` (main-spectroscopic-sample catalogue)

4. **Open the notebook** and update the path variables at the top (image directory, catalogue path, output directory).

5. **Run cells in order.** The notebook is sectioned so each model can be trained independently if desired.

Approximate runtime on a single T4 GPU (as used in development): about 2 hours end-to-end including the noise experiment.

---

## Data source and credits

Data from the **Galaxy Zoo 2** project:

- Lintott, C. et al., 2008. *Galaxy Zoo: Morphologies Derived from Visual Inspection of Galaxies from the Sloan Digital Sky Survey.* MNRAS 389(3): 1179–1189.
- Willett, K. W. et al., 2013. *Galaxy Zoo 2: Detailed Morphological Classifications for 304,122 Galaxies from the Sloan Digital Sky Survey.* MNRAS 435(4): 2835–2860.

Credit goes to the 100,000+ Galaxy Zoo volunteers who provided the underlying visual classifications. All imaging is from the Sloan Digital Sky Survey (SDSS).

---

## Citation

If you use this code or these results, please cite the MSc project report:

> Saraswatula, K. A. (2026). *Galaxy Morphology Classification using Deep Learning: A Comparative Study of Convolutional, Transfer-Learning and Transformer Approaches with Robustness and Interpretability Analysis.* MSc Data Science Project, University of Hertfordshire.

---

## Licence

Released under the MIT License — see [LICENSE](LICENSE).
