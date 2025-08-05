# Report on HOG, LBP, and Combined Feature Performance

**Including effect of Hard Negative Mining (Bootstrapping)**

> **Note:** The raw output data for all the runs and parameters used in this report can be found in the
> file [output.txt](output.txt) for reference.



---

## 1. Overview of Experiments

- Two main experiment sets:
    - **Normal Runs** (no hard negative mining)
    - **Bootstrapped Runs** (with hard negative mining)
- Tested parameters varied:
    - HOG bins (`nbins`): 6, 9, 12
    - Block size: [16x16], [24x24]
    - Block stride: [8x8], [16x16]
- Features tested:
    - HOG only
    - LBP only
    - BOTH (HOG + LBP combined)

---

## 2. Best Overall Results

### 2.1 Best Overall Accuracy & F1 Score (HOG only)

| Setting                                             | Accuracy   | Precision | Recall | F1 Score   |
|-----------------------------------------------------|------------|-----------|--------|------------|
| Bootstrapped, nbins=9, block=[16x16], stride=[8x8]  | **99.10%** | 96.86%    | 91.35% | **93.97%** |
| Bootstrapped, nbins=12, block=[16x16], stride=[8x8] | 99.02%     | 96.28%    | 90.88% | 93.49%     |
| Bootstrapped, nbins=12, block=[24x24], stride=[8x8] | 99.01%     | 96.53%    | 90.41% | 93.35%     |

**Interpretation:**

- The best accuracy and F1 scores come from the **bootstrapped runs** using **HOG with nbins=9 or 12**, block size 16x16
  or 24x24, and stride 8x8.
- Bootstrapping significantly improves accuracy and F1 compared to normal runs, reaching close to 99% accuracy and F1
  around 94%.

---

### 2.2 Best Overall Accuracy & F1 Score (LBP only)

| Setting                                            | Accuracy | Precision | Recall | F1 Score |
|----------------------------------------------------|----------|-----------|--------|----------|
| Bootstrapped, nbins=6, block=[16x16], stride=[8x8] | 94.88%   | 75.77%    | 49.47% | 59.78%   |

**Interpretation:**

- LBP alone achieves much lower recall and F1 scores compared to HOG.
- Bootstrapping improves LBP accuracy and precision but recall remains low (~50%), which drags F1 score down.

---

### 2.3 Best Overall Accuracy & F1 Score (Combined HOG+LBP)

| Setting                                             | Accuracy | Precision | Recall | F1 Score   |
|-----------------------------------------------------|----------|-----------|--------|------------|
| Bootstrapped, nbins=12, block=[16x16], stride=[8x8] | 98.31%   | 99.55%    | 78.48% | **87.76%** |
| Bootstrapped, nbins=12, block=[24x24], stride=[8x8] | 98.28%   | 99.56%    | 78.13% | 87.54%     |

**Interpretation:**

- Combined features with bootstrapping reach very high precision (~99.5%) and significantly better recall (~78%) than
  LBP alone.
- F1 scores are strong (~87.5-87.7%), but still below HOG alone (due to lower recall).

---

## 3. Effect of Hard Negative Mining (Bootstrapping)

- **Accuracy:** Increases by about 2-4% on average for HOG and combined features compared to normal runs.
- **Precision:** Noticeably improved in bootstrapped runs, often hitting above 95% for HOG, even nearing 99% in combined
  features.
- **Recall:** Slight improvements or stable for HOG, but remains limited for LBP.
- **F1 Score:** Bootstrapping yields clear improvements, especially for HOG and combined, pushing F1 from ~91% to ~94%
  in best cases.

**Summary:**  
Hard negative mining via bootstrapping significantly boosts the classifier’s precision and overall accuracy, with
moderate improvements in recall, resulting in better F1 scores. The combined feature set also benefits, especially in
precision, though recall stays moderate.

---

## 4. Impact of Parameters Individually

### 4.1 Number of HOG bins (`nbins`)

- Increasing `nbins` from 6 to 9 or 12 generally improves accuracy, precision, and F1 score, especially in bootstrapped
  runs.
- For example, bootstrapped runs with `nbins=9` or `12` outperform `nbins=6` consistently.

### 4.2 Block Size

- Larger block size `[24 x 24]` tends to slightly improve accuracy and precision over `[16 x 16]`.
- This effect is more pronounced in bootstrapped runs.

### 4.3 Block Stride

- Smaller stride `[8 x 8]` generally yields better performance than larger stride `[16 x 16]`, especially in accuracy
  and F1 score.
- The denser feature extraction with smaller stride seems to help.

---

## 5. Feature-wise Comparison Summary

| Feature Set    | Accuracy (best) | Precision (best) | Recall (best) | F1 Score (best) | Notes                                |
|----------------|-----------------|------------------|---------------|-----------------|--------------------------------------|
| HOG            | ~99.1% (boot)   | ~97% (boot)      | ~91% (boot)   | ~94% (boot)     | Best all-around performer            |
| LBP            | ~95.3% (boot)   | ~79.9% (boot)    | ~53% (boot)   | ~63.7% (boot)   | Low recall limits F1                 |
| BOTH (HOG+LBP) | ~98.3% (boot)   | ~99.5% (boot)    | ~78% (boot)   | ~87.7% (boot)   | Very high precision, recall moderate |

---

## 6. Conclussions

- **Use HOG features with `nbins=9 or 12`, block size `[16x16] or [24x24]`, and stride `[8x8]` combined with hard
  negative mining for best performance.**
- LBP features alone have limited recall and are best used to complement HOG rather than replace it.
- Hard negative mining is highly beneficial, especially for increasing precision and overall accuracy.
- Smaller stride values improve performance, likely due to denser feature representation.
- Combined features provide excellent precision but the recall needs improvement if higher sensitivity is needed.

---
