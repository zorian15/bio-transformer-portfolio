## Test-set results (annotated subset)

2534 held-out proteins from DeepLoc's homology-partitioned test split. Mean over 3 seeds, with standard deviation. Majority-class accuracy floor: 0.305.

| Arm | Purpose | Dim | Accuracy | Macro-F1 | Balanced acc. |
|---|---|---:|---:|---:|---:|
| sequence-only | baseline to beat | 480 | 0.757 ± 0.002 | 0.619 ± 0.007 | 0.602 ± 0.006 |
| text-only-free | how much does prose alone explain | 384 | 0.745 ± 0.006 | 0.664 ± 0.012 | 0.641 ± 0.005 |
| text-only-structured | leakage upper bound | 384 | 0.934 ± 0.002 | 0.911 ± 0.004 | 0.900 ± 0.006 |
| sequence+free-text | headline comparison | 864 | 0.845 ± 0.001 | 0.750 ± 0.003 | 0.723 ± 0.005 |
| sequence+structured | headline, with leaky text | 864 | 0.938 ± 0.002 | 0.903 ± 0.005 | 0.887 ± 0.004 |
| shuffled-text-control | detects gains not tied to this protein's text | 864 | 0.738 ± 0.005 | 0.588 ± 0.005 | 0.573 ± 0.005 |
