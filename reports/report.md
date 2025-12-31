# Fizyka Ogólna - projekt - Sprawozdanie

* Mikołaj Garbowski
* Maksym Bieńkowski

TODO

## Wyniki

```
=== Classification Report ===
              precision    recall  f1-score   support

     Klasa 6       0.46      0.50      0.48        22
    Klasa 15       0.63      0.72      0.67        68
    Klasa 16       0.88      0.88      0.88       139
    Klasa 42       0.45      0.45      0.45       183
    Klasa 52       0.00      0.00      0.00        29
    Klasa 53       0.00      0.00      0.00         6
    Klasa 62       0.53      0.28      0.37        85
    Klasa 64       0.39      0.60      0.47        15
    Klasa 65       0.88      0.96      0.92       135
    Klasa 67       0.00      0.00      0.00        31
    Klasa 88       0.96      0.96      0.96        56
    Klasa 90       0.68      0.83      0.75       338
    Klasa 92       0.86      0.82      0.84        38
    Klasa 95       0.90      0.84      0.87        32

    accuracy                           0.70      1177
   macro avg       0.54      0.56      0.55      1177
weighted avg       0.66      0.70      0.67      1177

=== Confusion Matrix ===
[[ 11   0   6   0   0   0   0   0   3   0   0   0   2   0]
 [  0  49   0   6   0   0   1   0   0   0   0  12   0   0]
 [  0   0 122   0   0   0   0   0  13   0   1   0   3   0]
 [  2  11   0  82   0   0  12   4   0   0   0  70   0   2]
 [  0   0   0  11   0   0   1   1   0   0   0  16   0   0]
 [  6   0   0   0   0   0   0   0   0   0   0   0   0   0]
 [  2   1   0  33   0   0  24   8   0   0   0  16   0   1]
 [  1   0   0   2   0   0   2   9   1   0   0   0   0   0]
 [  0   0   5   0   0   0   0   0 130   0   0   0   0   0]
 [  0   1   0   7   0   0   4   0   0   0   0  19   0   0]
 [  0   1   0   0   0   0   0   0   0   0  54   1   0   0]
 [  2  12   0  41   0   0   1   1   0   0   0 281   0   0]
 [  0   0   6   0   0   0   0   0   0   0   1   0  31   0]
 [  0   3   0   1   0   0   0   0   0   0   0   1   0  27]]
=== Micro-Averaged Metrics ===
Accuracy: 0.70
Precision: 0.70
Recall: 0.70
F1 Score: 0.70
=== Macro-Averaged Metrics ===
Accuracy: 0.56
Precision: 0.54
Recall: 0.56
F1 Score: 0.55
==============================
```

## Hiperparametry najlepszego modelu

```python
SupernovaClassifierV1Config(
    metadata_input_size=20, 
    metadata_num_hidden_layers=5, 
    metadata_hidden_size=128, 
    metadata_output_size=128, 
    lightcurve_input_size=6, 
    lightcurve_num_hidden_layers=4, 
    lightcurve_hidden_size=64, 
    classifier_hidden_size=512, 
    classifier_num_hidden_layers=3, 
    num_classes=14, 
    dropout=0.06791426425250689
)
```