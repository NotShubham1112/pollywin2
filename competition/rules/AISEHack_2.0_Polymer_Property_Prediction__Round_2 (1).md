**Title:** AISEHack 2.0 Polymer Property Prediction: Round 2

**Source:** [https://www.kaggle.com/competitions/ppp-round-2/data](https://www.kaggle.com/competitions/ppp-round-2/data)

---

# Page Structure Map
```text
AISEHack 2.0 Polymer Property Prediction: Round 2
├── Leaderboard and Qualification
├── Submission Format and Baseline
├── \- Generates predictions and creates a valid `submission.csv` file.
└── Files
    ├── **train.csv**
    ├── **test.csv**
    ├── **PI1M.csv**
    ├── **sample\_submission.csv**
    └── **baseline\_model.ipynb**
```

---

The training dataset contains **7,409 polymer property measurements** spanning **seven distinct polymer properties**. Each sample consists of a polymer represented by its **SMILES** string, the corresponding property value, and a **target\_type** indicating which property is being predicted. The seven target properties are:

1.  **Chain Bandgap (Egc)**
2.  **Bulk Bandgap (Egb)**
3.  **Ionisation Energy (Ei)**
4.  **Dielectric Constant (EPS)**
5.  **Electron Affinity (Eea)**
6.  **Refractive Index (Nc)**
7.  **Glass Transition Temperature (Tg)**

## Leaderboard and Qualification

Competition rankings will be based on predictions made on a **hidden private test set**, which serves as the official evaluation dataset. During the competition, participants will receive feedback through a **public leaderboard**, computed using a subset of the test data. The **final leaderboard** will be generated using the remaining hidden test samples and will determine the official rankings and qualification for the next stage.

## Submission Format and Baseline

A **sample submission file** is provided to demonstrate the required submission format. To help participants get started, we also provide a **baseline notebook** that demonstrates an end-to-end machine learning workflow. The baseline model:

-   Generates molecular descriptors from polymer SMILES using the **RDKit** library.
-   Performs basic feature engineering and preprocessing.
-   Trains a **Ridge Regression** model.

## \- Generates predictions and creates a valid `submission.csv` file.

## Files

### **train.csv**

Training dataset.

| Column | Description |
| --- | --- |
| `smiles` | SMILES representation of the polymer structure |
| `target` | Experimental value of one of the seven polymer properties |
| `target_type` | Property category corresponding to the target value |

### **test.csv**

Test dataset used for prediction, which has 4497 data points.

| Column | Description |
| --- | --- |
| `id` | Unique sample identifier |
| `smiles` | SMILES representation of the polymer structure |
| `target_type` | Property to be predicted |

### **PI1M.csv**

Additional polymer SMILES dataset that participants may use for implementing advanced algorithms.

| Column | Description |
| --- | --- |
| `SMILES` | Polymer SMILES strings |

### **sample\_submission.csv**

Example submission file illustrating the required prediction format.

### **baseline\_model.ipynb**

A baseline notebook demonstrating molecular descriptor generation using **RDKit**, feature preprocessing, Ridge Regression model training, and generation of a valid `submission.csv`.