| Training | Testing| Optimizer | lr | bs |Accuracy (%) | MIoU(%) | Training Time |
|----------|----------|----------|----------|----------|----------|----------|----------|
| Cityscapes | Cityscapes |Adam| 0.001 | 8 | 81.0 | 54.2 | 03:03|
| GTA5 | GTA5 |Adam| 0.001 | 8 | 82.8 | 64.7 | 02:43 |
| GTA5 | Cityscapes |-| 0.001 | 8 | 49.0 | 16.5 |-|
| GTA5 (Data Augmentation) | Cityscapes |Adam| 0.001 | 8 |69.1  | 26.0 | 02:59 | 
| GTA5 (Adversarial Domain Adaptation) | Cityscapes |SGD| 0.001 | 8 |68.6  | 26.1 | 04:54 | 
| GTA5 (Adversarial Domain Adaptation and FDA beta 0.01 ) | Cityscapes |SGD| 0.001 | 8 |64.7  | 22.2 | 05:20 | 
| GTA5 (Adversarial Domain Adaptation and FDA beta 0.05 ) | Cityscapes |SGD| 0.001 | 8 |67.4  | 23.6 | 05:17 | 
| GTA5 (Adversarial Domain Adaptation and FDA beta 0.09 ) | Cityscapes |SGD| 0.001 | 8 | 64.9 | 22.0 | 05:18 | 
| GTA5 (Adversarial Domain Adaptation and FDA all betas ) | Cityscapes |-| 0.001 | 8 |69.3  | 25.2 | - | 
| GTA5 (pseudo label and FDA beta 0.05 - 10 epoch ) | Cityscapes |SGD| 0.001 | 8 |69.3  | 25.3 | 05:48 | 