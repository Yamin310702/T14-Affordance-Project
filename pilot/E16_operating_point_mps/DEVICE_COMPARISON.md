# E16 CPU–MPS comparison

Both runs use the same fixed model, calibration object IDs, fresh test object IDs, query wording, and scoring rule. E16 CPU and MPS selected the same threshold (1.00) for both query variants. At that cutoff both predicted empty maps and detected 0/20 positive objects, with 0/20 negative-object alarms.

| Score-map group | Maps | Maximum absolute CPU–MPS difference | Mean absolute difference | allclose at atol=rtol=1e-5 |
|---|---:|---:|---:|---|
| Fresh test, original prompt | 40 | 6.29×10⁻⁶ | 3.22×10⁻⁷ | Yes |
| Fresh test, selected paraphrase | 40 | 2.44×10⁻⁶ | 2.57×10⁻⁷ | Yes |
| Calibration, original prompt | 80 | 0 | 0 | Yes |
| Calibration, selected paraphrase | 80 | 2.80×10⁻⁶ | 2.61×10⁻⁷ | Yes |

Threshold-averaged aIoU, AUC, SIM, MAE, and the diagnostic 0.5 alarm counts matched to the displayed precision. These differences are negligible for this experiment; MPS did not change its conclusion.
