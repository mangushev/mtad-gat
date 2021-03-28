# mtad-tf
Implementation of MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Network

https://arxiv.org/pdf/2009.02040.pdf


This is a work in progress

Notes:

- both feature & temporal GATs are tested with forecasting. It seems to perform better compare to mtad-tf. But this is just preliminary

- I found training is quite tricky. Temporal GAT sigmoid produces 0.5 or 1. I solved it only with low training rate and clipping gradients

- I did VAE but I do not see how bottleneck can be 300. Also, fprecasting loss is MSR, but VAE reconstruction is pdf and so loss values are very different.
