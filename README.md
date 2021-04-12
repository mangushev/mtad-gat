# mtad-gat
Implementation of MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Network

https://arxiv.org/pdf/2009.02040.pdf

Other papers

N/A

Notes:

this is draft. both forecasting and reconstraction modes are trainable separately. forecasting only gives good results, reconstruction only is not as good as i see for now. combined training works, but i did not do it yet.

- run_mode flag to specify to train or predict in FORECASTING, RECONSTRUCTING or BOTH . if mode is trained in FORECASTING , it does not make any sense to predict other than FORECASTING and same about other modes

- clip gradients is set to 0.1 and learning rate 5e-6

- d3 is 18, it is kind of half of 38 features what is of SMD

- reconstruction pdf taken and used for every time point in a time series, not just last one

- if reconstruction, anomaly log pdf is -reconstruction log pdf. it is not 1 - pdf

- to calculate combined score , anomaly log pdf is calculated as explaned above. both forecasting  square difference and reconstruction anomaly pdf are scaled to a range 0-1 and features are summed. gamma can be used to give more weight to forecasting or reconstracting. combined score can be used to show which feature caused anomaly the most. 

- final score is total of all feature scores. this is compared against threshold

- train and test data is generated to GCP

- models are stored to GCP

- "anomaly_detection" is my GCP storage

- I used some very minor utility content from BERT. This is why I put BERT Licence

- I am discussing SMD only, but MSL and SMAP should be used as well


Training steps:

1. Download ServerMachineDataset from https://github.com/NetManAIOps/OmniAnomaly
Put ServerMachineDataset here at the root of the 

2. Prepare tfrecords train and test data

python prepare_data.py --files_path=ServerMachineDataset/train --tfrecords_file=gs://anomaly_detection/mtad_tf/data/train/{}.tfrecords
python prepare_data.py --files_path=ServerMachineDatasettest --label_path=ServerMachineDataset/test_label --tfrecords_file=gs://anomaly_detection/mtad_tf/data/test/{}.tfrecords

3. Decide the mode of the model. Train model for each machine dataset, 28 of them. Farecasting seems 30k steps works. RECONSTRUCTING - it took 300k. BOTH, I did not do it yet. Models sometimes deviate to NaN during backpropagation, so I just restart it.

for c in {1..100}; do echo $c; time python training.py --action=TRAIN --train_file=gs://anomaly_detection/mtad_gat/data/train/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_gat/output/machine-1-1-fore --run_mode=FORECASTING --num_train_steps=30000; done

for c in {1..100}; do echo $c; time python training.py --action=TRAIN --train_file=gs://anomaly_detection/mtad_gat/data/train/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_gat/output/machine-1-1-reco --run_mode=RECONSTRACTING --num_train_steps=300000; done

for c in {1..100}; do echo $c; time python training.py --action=TRAIN --train_file=gs://anomaly_detection/mtad_gat/data/train/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_gat/output/machine-1-1-both --run_mode=BOTH --num_train_steps=300000; done

4. Produce scores file - inference_score.csv. Use this file to calculate calculate threshold using POT

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_gat/output/machine-1-1-fore --prediction_task=inference_score --run_mode=FORECASTING

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_gat/output/machine-1-1-fore --prediction_task=inference_score --run_mode=RECONSTRACTING

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_gat/output/machine-1-1-fore --prediction_task=inference_score --run_mode=BOTH

5. Use my another repository EVT_POT to calculate threshold. inference_score.csv is an input for that

6. Use notebook to grath inference_score.csv, initial threshold, POT and Best threshold. Adjust those values in the notebook


Anomaly Evaluation

I do not use Estimator EVAL to do this since I use an adjustment procedure. I use this same simple omnianomaly approach. If we predicted anomaly and it is within some anomaly segment, whole segment becomes correctly predicted. I do not use a prediction latency (delta) value as in some other papers. Also, I think practically, if we flagged anomaly earlier compare to actual anomaly label, it could be considered hit maybe using some delta as well. I did not do this way here. Please see samples folder for evaluations for machine dataset 1-1, 1-2 and 1-3

You must provide your threshold values

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1-fore --threshold=1.5547085 --prediction_task=EVALUATE --run_mode=FORECASTING

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1-reco --threshold=1.5547085 --prediction_task=EVALUATE --run_mode=RECONSTRACTING

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1-both --threshold=1.5547085 --prediction_task=EVALUATE --run_mode=BOTH

Anomaly Prediction

This command creates Anomaly.csv in the local folder. You must provide your threshold values

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1-fore --threshold=1.5547085 --run_mode=FORECASTING

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1-fore --threshold=1.5547085 --run_mode=RECONSTRACTING

python training.py --action=PREDICT --test_file=gs://anomaly_detection/mtad_gat/data/test/machine-1-1.tfrecords --output_dir=gs://anomaly_detection/mtad_tf/output/machine-1-1-fore --threshold=1.5547085 --run_mode=BOTH
