# KL attribution, Qwen3.5-2B, solved blocks: gptq

Mean KL against the F16 base (WikiText-2, 16 x 512). 'only' quantizes the class and keeps the
rest F16. 'except' quantizes everything but the class. The full file is the last row.

| Class | only: mean KL | only: top-1 | except: mean KL | except: top-1 |
|---|---|---|---|---|
| head | 0.011325 | 92.819 % | 0.026266 | 90.760 % |
| embedding | 0.000367 | 98.873 % | 0.037323 | 88.676 % |
| attention k/v (Q8) | 0.000376 | 98.897 % | 0.037195 | 88.652 % |
| attention q | 0.000879 | 98.235 % | 0.036800 | 88.995 % |
| attention o | 0.001576 | 97.574 % | 0.035895 | 88.946 % |
| GDN qkv | 0.002536 | 97.132 % | 0.035171 | 89.240 % |
| GDN gate z | 0.001604 | 97.745 % | 0.035602 | 88.946 % |
| GDN out | 0.004818 | 95.809 % | 0.032522 | 89.436 % |
| MLP gate/up | 0.008801 | 94.289 % | 0.028017 | 89.681 % |
| MLP down | 0.007014 | 95.490 % | 0.030355 | 89.436 % |
| layers 0-2 | 0.007469 | 95.637 % | 0.030386 | 89.314 % |
| layers 21-23 | 0.004721 | 96.201 % | 0.032623 | 89.436 % |
| all | 0.037305 | 88.676 % | | |
