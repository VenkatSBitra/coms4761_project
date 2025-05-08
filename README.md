# ArcCell: A Robust and Generalizable Single Cell RNA-seq Cell Type Annotation Tool

## COMS 4761 Computational Genomics - Final Project

This repository contains the code and information related to the final project for COMS 4761 Computational Genomics, focusing on ArcCell, a lightweight and generalizable model for single-cell RNA-seq cell type annotation. ArcCell combines balanced preprocessing with ArcFace-based dimensionality reduction and a supervised classifier, achieving strong performance across diverse datasets. [cite: 1, 98]

## Abstract

We introduce ArcCell, a lightweight and generalizable model for single-cell RNA-seq cell type annotation. ArcCell combines balanced preprocessing with ArcFace-based dimensionality reduction and a supervised classifier, achieving strong performance across diverse datasets. On mouse gonadal data, ArcCell reached a macro F1 score of 0.96 on the test set and 0.94 on out-of-sample data, substantially outperforming the CellTypist baseline (F1 scores of 0.62). [cite: 1] The model is especially effective for rare cell types, demonstrating high precision and recall. These results highlight ArcCell's robustness and potential for scalable, accurate cell annotation in biological research. [cite: 1]

## Data

The datasets used for this project are for male and female mouse gonadal cells:

* **Male Mouse Gonadal Data:** [https://datasets.cellxgene.cziscience.com/c3f6f3c6-2831-46fe-9d89-34d3f39ed5a6.h5ad](https://datasets.cellxgene.cziscience.com/c3f6f3c6-2831-46fe-9d89-34d3f39ed5a6.h5ad)
* **Female Mouse Gonadal Data:** [https://datasets.cellxgene.cziscience.com/51fb2153-0522-480e-aa8c-e59b1d75896f.h5ad](https://datasets.cellxgene.cziscience.com/51fb2153-0522-480e-aa8c-e59b1d75896f.h5ad)

Download these files and place them in a `data/` directory within the project.

## Code Files and Usage

The following Python scripts are used in this project:

* **`preprocess.py`**: This script is used for preprocessing the raw data. It includes steps such as filtering cells and genes, normalization, and log transformation. It prepares the data for input into the models.
* **`dnn.py`**: This script implements the ArcCell model. ArcCell utilizes an ArcFace-based dimensionality reduction technique followed by a supervised classifier. [cite: 1]
* **`celltypist.py`**: This script implements the CellTypist model with an unbalanced dataset approach, serving as a baseline for comparison.
* **`celltypist_balanced.py`**: This script implements the CellTypist model with a balanced dataset approach.

**General Workflow:**

1.  **Download Data:** Obtain the male and female mouse gonadal `.h5ad` files from the links above and save them in a `data/` directory.
2.  **Preprocess Data:** Run `preprocess.py` to clean and prepare the datasets. This will likely create output files in a `cleaned_data/` directory.
    ```bash
    python preprocess.py
    ```
3.  **Run Models:**
    * To run the ArcCell model:
        ```bash
        python dnn.py
        ```
    * To run the unbalanced CellTypist model:
        ```bash
        python celltypist.py
        ```
    * To run the balanced CellTypist model:
        ```bash
        python celltypist_balanced.py
        ```

## Models

### ArcCell

ArcCell is a novel cell annotation model that integrates preprocessing techniques to balance rare and common cell types with a classifier that uses a highly discriminative dimensionality reduction method (ArcFace). [cite: 102, 103] This approach aims to annotate cell types effectively and efficiently. [cite: 103] A schematic of the ArcCell pipeline is illustrated below:

*Input -> Normalized Embedding -> ArcFace Weights -> Cos(θyi) -> arccos(cos(θyi)) = θyi -> θyi + m -> cos(θyi + m) -> Rescaled Feature Vector -> Softmax Probability -> Ground Truth -> Cross-entropy loss*
*Input -> Normalized Embedding -> ArcFace Weights -> Cos(θyi) -> Softmax Probability -> Ground Truth -> Cross-entropy loss*
[cite: 66]

### CellTypist (Baseline)

CellTypist is used as a baseline model for comparison. The project evaluates both unbalanced and balanced approaches for CellTypist.

## Results

The ArcCell pipeline demonstrated significant improvements in cell type annotation accuracy compared to the CellTypist baseline across multiple evaluation metrics. [cite: 69]

* **Male Mouse Gonadal Test Dataset (33120 cells):**
    * ArcCell: Accuracy 95.73%, F1 Score 0.9573 [cite: 70]
    * CellTypist: Accuracy 58.71%, F1 Score 0.6224 [cite: 70]
* **Female Mouse Gonadal Dataset (Out-of-sample, 33120 cells):**
    * ArcCell: Accuracy 93.51%, F1 Score 0.9375 [cite: 71]
    * CellTypist: Accuracy 60.71%, F1 Score 0.6192 [cite: 71]

ArcCell showed particular strength in classifying rare cell types like erythrocytes (94% precision and 94% recall for ArcCell vs. 12% precision and 99% recall for CellTypist on the female dataset). [cite: 72, 89] For mesenchymal cells, the most abundant type, ArcCell achieved 98% precision and 89% recall on the female dataset, compared to CellTypist's 100% precision but only 21% recall. [cite: 73, 89]

**Comparative Results on Test Dataset (Male Gonadal):** [cite: 86, 87]

| Cell Type            | ArcCell Precision | ArcCell Recall | ArcCell F1-Score | CellTypist Precision | CellTypist Recall | CellTypist F1-Score |
| -------------------- | ----------------- | -------------- | ---------------- | -------------------- | ----------------- | ------------------- |
| Endothelial cell     | 0.98              | 1.00           | 0.99             | 0.43                 | 1.00              | 0.60                |
| Epithelial cell      | 0.93              | 0.94           | 0.93             | 0.84                 | 0.75              | 0.69                |
| Erythrocyte          | 1.00              | 0.80           | 0.88             | 0.07                 | 0.13              | 0.09                |
| Germ cell            | 1.00              | 0.99           | 1.00             | 1.00                 | 0.90              | 1.00                |
| Mesenchymal cell     | 0.97              | 0.96           | 0.97             | 1.00                 | 0.53              | 0.37                |
| Neural cell          | 0.90              | 1.00           | 1.00             | 0.80                 | 0.60              | 1.00                |
| Skeletal muscle fiber| 0.97              | 0.98           | 0.97             | 0.34                 | 0.29              | 0.99                |
| Supporting cell      | 0.98              | 0.91           | 0.92             | 0.41                 | 0.58              | 1.00                |
| **Macro Avg** | **0.97** | **0.97** | **0.97** | **0.60** | **0.56** | **0.61** |
| **Weighted Avg** | **0.96** | **0.96** | **0.96** | **0.86** | **0.59** | **0.62** |
| **Overall Accuracy** |                   |                | **0.9573** |                      |                   | **0.5871** |
| **Overall F1 Score** |                   |                | **0.9573** |                      |                   | **0.6224** |

**Comparative Results on Out-of-Sample Dataset (Female Gonadal):** [cite: 88, 89]

| Cell Type            | ArcCell Precision | ArcCell Recall | ArcCell F1-Score | CellTypist Precision | CellTypist Recall | CellTypist F1-Score |
| -------------------- | ----------------- | -------------- | ---------------- | -------------------- | ----------------- | ------------------- |
| Endothelial cell     | 0.98              | 0.90           | 0.99             | 0.73                 | 1.00              | 0.84                |
| Epithelial cell      | 0.83              | 0.98           | 0.90             | 0.77                 | 0.80              | 0.79                |
| Erythrocyte          | 0.94              | 0.94           | 0.94             | 0.12                 | 0.99              | 0.21                |
| Germ cell            | 1.00              | 0.98           | 0.99             | 1.00                 | 0.95              | 1.00                |
| Mesenchymal cell     | 0.98              | 0.89           | 0.93             | 1.00                 | 0.21              | 0.34                |
| Neural cell          | 0.89              | 1.00           | 0.94             | 0.93                 | 0.99              | 0.96                |
| Skeletal muscle fiber| 0.57              | 0.93           | 0.71             | 0.18                 | 0.90              | 0.31                |
| Supporting cell      | 0.98              | 0.95           | 0.97             | 0.80                 | 0.58              | 0.67                |
| **Macro Avg** | **0.90** | **0.96** | **0.92** | **0.70** | **0.85** | **0.66** |
| **Weighted Avg** | **0.95** | **0.94** | **0.94** | **0.89** | **0.61** | **0.62** |
| **Overall Accuracy** |                   |                | **0.9351** |                      |                   | **0.6071** |
| **Overall F1 Score** |                   |                | **0.9375** |                      |                   | **0.6192** |

t-SNE visualizations also revealed that ArcCell produced tighter, more biologically coherent clusters compared to baseline methods. [cite: 76] Marker gene analysis demonstrated ArcCell's ability to identify biologically relevant marker genes, matching the baseline in identifying canonical markers and excelling at uncovering genes defining rare or transitional cell populations. [cite: 81, 94]

## Authors

* Venkat Suprabath Bitra
* Rika Chan
* Kristi Xing

## Acknowledgments

The authors thank the class, Professor Pe'er, and the TAs for their invaluable feedback on the outline, midterm, and final presentations. [cite: 107]

## References
(See Genomics_Final_Report.pdf for a full list of references) [cite: 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]
