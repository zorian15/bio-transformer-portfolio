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
| text-only-free-cleaned | prose alone, database bookkeeping removed | 384 | 0.702 ± 0.003 | 0.630 ± 0.006 | 0.600 ± 0.005 |
| text-only-free-ablated | prose alone, compartment sentences removed | 384 | 0.639 ± 0.001 | 0.482 ± 0.008 | 0.468 ± 0.007 |
| text-only-free-random-ablated | prose alone, as much text removed at random | 384 | 0.653 ± 0.004 | 0.518 ± 0.011 | 0.496 ± 0.007 |
| sequence+free-text-cleaned | isolates the evidence-code confound | 864 | 0.842 ± 0.003 | 0.743 ± 0.003 | 0.718 ± 0.001 |
| sequence+free-text-ablated | the ablation: grounding or leakage | 864 | 0.807 ± 0.008 | 0.656 ± 0.013 | 0.638 ± 0.011 |
| sequence+free-text-random-ablated | length-matched control for the ablation | 864 | 0.815 ± 0.009 | 0.671 ± 0.017 | 0.651 ± 0.015 |
