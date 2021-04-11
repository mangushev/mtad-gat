# mtad-gat
Implementation of MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Network

https://arxiv.org/pdf/2009.02040.pdf


this is draft. both forecasting and reconstraction modes are trainable separately. forecasting only gives good results, reconstruction only is not as good as i see for now. combined training works, but i did not do it yet.

Notes:

- run_mode flag to specify to train or predict in FORECASTING, RECONSTRUCTING or BOTH . if mode is trained in FORECASTING , it does not make any sense to predict other than FORECASTING and same about other modes

- clip gradients is set to 0.1 and learning rate 5e-6

- d3 is 18 kind of half of 38 features I use in computer data set

- reconstruction pdf taken and used for every time point in a time series, not just last one

- if reconstruction, anomaly log pdf is -reconstruction log pdf. it is not 1 - pdf

- to calculate combined score , anomaly log pdf is calculated as explaned above. both forecasting  square difference and reconstruction anomaly pdf are scaled to a range 0-1 and features are summed. gamma can be used to give more weight to forecasting or reconstracting. combined score can be used to show which feature caused anomaly the most. 

- final score is total of all feature scores. this is compared against threshold
