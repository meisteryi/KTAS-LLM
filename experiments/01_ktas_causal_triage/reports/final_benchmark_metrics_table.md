| Model                         | Split               |   DRA (%) |   OIR (%) |   CSR-Down (%) |   CSR-M (%) |   RTAS (%) |
|:------------------------------|:--------------------|----------:|----------:|---------------:|------------:|-----------:|
| Qwen2.5-32B (Zero-Shot)       | In-Distribution     |      58.2 |      62.4 |           71   |        68.5 |       54   |
| Qwen2.5-32B (Zero-Shot)       | Out-of-Distribution |      49.5 |      55.1 |           64.2 |        60   |       48.2 |
| Qwen2.5-32B (Few-Shot)        | In-Distribution     |      74.5 |      78.1 |           82.3 |        80   |       71.2 |
| Qwen2.5-32B (Few-Shot)        | Out-of-Distribution |      66.8 |      71.3 |           75   |        73.5 |       63.8 |
| Qwen2.5-32B (LoRA SFT - Ours) | In-Distribution     |      95.5 |      99   |           97.2 |        98   |       96.5 |
| Qwen2.5-32B (LoRA SFT - Ours) | Out-of-Distribution |      88.4 |      96.2 |           92.5 |        94.1 |       91   |