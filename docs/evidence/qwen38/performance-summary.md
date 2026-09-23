# Repeated workload results

TTFT is seconds; SLO requires successful completion and arrival-to-first-token within the preset threshold.
Output throughput includes partial output; completed-request throughput excludes aborted/failed requests.

| Scenario | Variant | VIP n | TTFT p50 / p95 / p99 (s) | VIP SLO | Ordinary complete | Output / completed tok/s |
| --- | --- | ---: | --- | --- | --- | ---: |
| burst | abort | 64 | 0.2482 / 0.2708 / 0.2772 | 100.0% | 50.0% | 62.16 / 60.64 |
| burst | native | 64 | 31.7064 / 32.3106 / 32.6158 | 0.0% | 100.0% | 62.91 / 62.91 |
| burst | native-async | 64 | 31.8283 / 32.4301 / 32.6147 | 0.0% | 100.0% | 63.00 / 63.00 |
| burst | recompute | 64 | 0.2478 / 0.2680 / 0.2719 | 100.0% | 100.0% | 62.42 / 62.42 |
| long-prefill | abort | 32 | 0.4114 / 0.4428 / 0.4443 | 100.0% | 50.0% | 35.08 / 34.84 |
| long-prefill | native | 32 | 32.4523 / 32.9652 / 33.0763 | 0.0% | 100.0% | 58.60 / 58.60 |
| long-prefill | native-async | 32 | 32.4381 / 32.9398 / 33.0893 | 0.0% | 100.0% | 58.77 / 58.77 |
| long-prefill | recompute | 32 | 0.4140 / 0.4444 / 0.4496 | 100.0% | 100.0% | 57.78 / 57.78 |
| saturated | abort | 32 | 0.2502 / 0.2724 / 0.2752 | 100.0% | 50.0% | 35.89 / 35.24 |
| saturated | native | 32 | 31.9048 / 32.3633 / 32.5307 | 0.0% | 100.0% | 59.44 / 59.44 |
| saturated | native-async | 32 | 32.0027 / 32.4314 / 32.6167 | 0.0% | 100.0% | 59.52 / 59.52 |
| saturated | recompute | 32 | 0.2328 / 0.2687 / 0.2706 | 100.0% | 100.0% | 59.12 / 59.12 |
| spare | abort | 16 | 0.0949 / 0.1066 / 0.1066 | 100.0% | 100.0% | 20.01 / 20.01 |
| spare | native | 16 | 0.0921 / 0.1046 / 0.1046 | 100.0% | 100.0% | 20.08 / 20.08 |
| spare | native-async | 16 | 0.1560 / 0.2018 / 0.2018 | 100.0% | 100.0% | 20.18 / 20.18 |
| spare | recompute | 16 | 0.0933 / 0.1018 / 0.1018 | 100.0% | 100.0% | 20.04 / 20.04 |

Quantiles use nearest rank. Small-sample tails are descriptive. Paired bootstrap intervals and all request metrics are in the JSON.
