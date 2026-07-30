## Test-set results (all proteins)

2773 held-out proteins from DeepLoc's homology-partitioned test split. Mean over 3 seeds, with standard deviation. Majority-class accuracy floor: 0.291.

| Arm | Purpose | Dim | Accuracy | Macro-F1 | Balanced acc. |
|---|---|---:|---:|---:|---:|
| sequence-only | baseline to beat | 480 | 0.755 ± 0.004 | 0.616 ± 0.004 | 0.599 ± 0.004 |
| text-only-free | how much does prose alone explain | 384 | 0.690 ± 0.005 | 0.617 ± 0.015 | 0.585 ± 0.010 |
| text-only-structured | leakage upper bound | 384 | 0.936 ± 0.001 | 0.912 ± 0.001 | 0.899 ± 0.003 |
| sequence+free-text | headline comparison | 864 | 0.835 ± 0.007 | 0.740 ± 0.006 | 0.716 ± 0.009 |
| sequence+structured | headline, with leaky text | 864 | 0.939 ± 0.001 | 0.906 ± 0.005 | 0.893 ± 0.005 |
| shuffled-text-control | detects gains not tied to this protein's text | 864 | 0.737 ± 0.002 | 0.578 ± 0.006 | 0.564 ± 0.004 |
