# NSAI-Humaine User Guide

NSAI-Humaine is a no-code interface for training **Logic Tensor Networks (LTN)** — neural networks that integrate user-defined logical rules. It supports tabular, image, and text data, and provides explainable predictions via SHAP and rule satisfaction analysis.

---

## Getting Started

### Docker (recommended)

```bash
docker-compose up
```

- Streamlit UI: http://localhost:8888
- JupyterLab (custom training): http://localhost:8889

### Python

```bash
pip install -r requirements.txt -r requirements2.txt -r requirements3.txt
streamlit run main.py
```

---

## Workflow Overview

The app has four tabs that guide you through the full pipeline:

```
Load → Predicates → Train → Explain
```

---

## Tab 1: Load Data

Choose your data source and let the app prepare it for training.

### CSV

Upload a CSV file directly. The app reads it as-is — each row is a sample, each column is a feature.

### Image Dataset

Upload a ZIP file containing one subfolder per class:

```
dataset.zip
├── class_0/
│   ├── image1.jpg
│   └── image2.png
└── class_1/
    ├── image3.jpg
    └── image4.png
```

The app extracts two types of features from each image:

- **Semantic tags** — uses the OpenAI Vision API to describe image contents (e.g., `has_irregular_border`, `high_contrast`). Similar tags are merged automatically.
- **Texture features** — numerical statistics (contrast, homogeneity, etc.) computed via GLCM.

Both are combined into a tabular format for training. Extraction results are cached — reloading the same dataset skips re-extraction unless you enable **Force Refresh**.

> Requires an OpenAI API key set in the environment (`OPENAI_API_KEY`).

### Text Dataset

Upload a CSV, TSV, or TXT file with at least two columns: one for the raw text and one for the label.

The app sends text samples to the OpenAI API in batches to extract discriminative semantic tags (e.g., `contains_link`, `urgent_tone`). Each tag becomes a boolean column. Similar tags are merged using semantic similarity.

Configurable options:

| Option | Description |
|--------|-------------|
| Batch size | Number of samples per API call (5–250) |
| Max tags | Number of top discriminative tags to keep (default: 30) |

> Requires an OpenAI API key set in the environment (`OPENAI_API_KEY`).

### MinIO

Enter a MinIO path to load data from cloud storage.

---

### After Selecting a Data Source

Once data is loaded, configure:

- **Dataset context** *(optional)*: Describe the domain (e.g., "medical breast cancer biopsy"), what a row represents, and what the model should predict. This improves the quality of generated explanations.
- **Target column**: The column the model should predict.
- **Columns to drop**: Remove irrelevant or leaky columns.

Click **Apply** to finalize.

---

## Tab 2: Predicates

Predicates are named logical conditions on your features. They form the building blocks of the rules you write in the next step.

### Simple Predicates

Select a feature column and define a threshold condition:

| Feature type | Example predicate |
|--------------|-------------------|
| Numeric | `high_clump_thickness` → column_5 > 3.0 |
| Boolean (0/1) | `has_irregular_border` → image tag column |

Give the predicate a meaningful name — you will reference it by name when writing rules.

### Composite Predicates

Combine existing predicates using `AND`, `OR`, or `NOT`:

```
suspicious = AND(high_clump_thickness, NOT(normal_mitosis))
```

Predicate code is auto-generated and saved to `output/predicates.txt`.

---

## Tab 3: Train

### Step 1 — Write Rules

Enter logical rules in plain English using the predicate names you defined:

```
if high_clump_thickness and high_mitoses then malignant
if not suspicious_growth then benign
```

Each rule becomes an LTN axiom. The model trains by maximizing how well these rules are satisfied across the training data. Save rules when done — they are parsed into `output/ltn_rules.txt`.

### Step 2 — Choose Training Method

**Automated**

Click **Train Models** to run the full training pipeline directly in the app. This trains three models:

| Model | Description |
|-------|-------------|
| LTN | Logic Tensor Network with your rules |
| MLP | Baseline neural network without rules |
| Rules-only | Pure logical reasoning (no learned weights) |

**Custom (JupyterLab)**

Click **Open Notebook** to open JupyterLab with `notebooks/ltn_training_session.ipynb`. Use this to tune hyperparameters (learning rate, epochs, batch size, etc.) and re-run training manually. The notebook writes model files to `output/` the same way the automated path does.

### Step 3 — Evaluate

Click **Evaluate Models** to see a comparison table:

| Metric | LTN | MLP | Rules-only |
|--------|-----|-----|------------|
| Accuracy | ... | ... | ... |
| F1 | ... | ... | ... |
| Precision | ... | ... | ... |
| Recall | ... | ... | ... |
| AUROC | ... | ... | ... |

---

## Tab 4: Explain

Select a test sample to inspect a specific prediction.

- **Original data**: For image datasets, the source image is displayed. For text datasets, the raw text is shown.
- **Feature values**: The processed feature vector used as model input.

Click **Predict!** to see:

- **Predictions**: Scores from all three models side by side.
- **Feature importance**: SHAP decision plot showing which features pushed the prediction toward each class.
- **Rules triggered**: Which of your logical rules were satisfied for this sample.
- **Natural language summary**: A RAG-generated explanation combining the triggered rules, important features, and dataset domain context.

---

## Output Files

All outputs are written to the `output/` directory:

| File | Contents |
|------|----------|
| `ltn.h5` | Trained LTN model weights |
| `mlp.h5` | Trained MLP model weights |
| `predicates.txt` | Auto-generated predicate code |
| `ltn_rules.txt` | Parsed LTN axioms from your rules |
| `rules_raw.txt` | Your original natural language rules |
| `feature_names.txt` | List of all feature columns used |

---

## Example: Breast Cancer (CSV)

1. **Load** `input/breast_cancer.csv`, set target = `class`, drop `id`.
2. **Predicates** — define `high_clump_thickness` (column_1 > 3), `high_mitoses` (column_8 > 2).
3. **Rules** — write `if high_clump_thickness and high_mitoses then malignant`.
4. **Train** — click Train Models, then Evaluate.
5. **Explain** — select a test sample, click Predict, review SHAP plot and triggered rules.

## Example: Brain Tumor (Images)

1. **Load** — upload `input/brain_tumor_images.zip`, set target = `label`.
2. The app extracts semantic tags (e.g., `irregular_mass`, `high_contrast_region`) and texture features automatically.
3. **Predicates** — define predicates on extracted tag columns.
4. **Rules** — write rules using those predicates.
5. **Train**, **Explain** as above. The Explain tab shows the original MRI alongside the prediction.

## Example: SMS Spam (Text)

1. **Load** — upload `input/sms_spam/` dataset as a CSV, set text column = `text`, target = `label`.
2. The app extracts tags like `contains_link`, `urgent_tone`, `prize_claim` via the OpenAI API.
3. **Predicates** — define `is_promotional` = contains_link AND prize_claim.
4. **Rules** — write `if is_promotional and urgent_tone then spam`.
5. **Train**, **Explain** as above. The Explain tab shows the original SMS text.
