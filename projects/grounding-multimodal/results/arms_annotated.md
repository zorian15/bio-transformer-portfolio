## Test-set results (annotated subset)

2534 held-out proteins from DeepLoc's homology-partitioned test split. Mean over 3 seeds, with standard deviation. Majority-class accuracy floor: 0.305.

| Arm | Purpose | Dim | Accuracy | Macro-F1 | Balanced acc. |
|---|---|---:|---:|---:|---:|
| sequence-only | baseline to beat | 480 | 0.758 ± 0.002 | 0.623 ± 0.007 | 0.605 ± 0.005 |
| text-only-free | how much does prose alone explain | 384 | 0.745 ± 0.006 | 0.664 ± 0.012 | 0.641 ± 0.005 |
| text-only-structured | leakage upper bound | 384 | 0.934 ± 0.002 | 0.911 ± 0.004 | 0.900 ± 0.006 |
| sequence+free-text | headline comparison | 864 | 0.845 ± 0.001 | 0.749 ± 0.003 | 0.722 ± 0.006 |
| sequence+structured | headline, with leaky text | 864 | 0.937 ± 0.000 | 0.901 ± 0.002 | 0.885 ± 0.000 |
| shuffled-text-control | detects gains not tied to this protein's text | 864 | 0.738 ± 0.006 | 0.588 ± 0.004 | 0.573 ± 0.004 |
