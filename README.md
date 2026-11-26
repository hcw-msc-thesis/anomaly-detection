This repository contains an anomaly detection model for the MVTec 3D-AD dataset for the thesis of Henry Wilson for MSc Big Data Analytics

Please ensure you have the 3DAD dataset downloaded and extracted.
To use the Grafana dashboard, ensure you have Docker installed and also the official MVTec 3D-AD evaluation scripts.

Preprocessing:

1. Use the command "python utils/preprocessing.py ./***ROOT_OF_MVTEC***" - this removes the background from the .tiffs and pads them to be square. Note, this can take a very long time for all of the data
2. Use the command "python crop_rotate_resize.py --data_root ./***ROOT_OF_MVTEC***" - this rotates, crops and resizes the files to be able to be used in the training and evaluation scripts

Training

1. Run the train_unet.py script with "python train_unet.py --data_root ./***ROOT_OF_PREPROCESSED_DATA*** --category ***category***" - this will train the unet on the preprocessed rgb data for the category of choice
2. Run the train_fpfh.py script with "python train_fpfh.py --data_root ./***ROOT_OF_PREPROCESSED_DATA*** --category ***category***" - this extracts the features from the preprocessed point clouds

It's recommended that all the categories are trained first before proceeding to the next step

Heatmap Generation

1. Run the generate_heatmaps.py script with "python generate_heatmaps.py --data_root "./***ROOT_OF_PREPROCESSED_DATA*** - this will create the heatmaps for each test sample and output them to ./heatmaps
2. Follow the instructions in the MVTec 3DAD evaluation scripts to produce the AUROC and AUPRO scores for the resulting heatmaps

Grafana Deliverable - ensure Docker is installed and running

1. Using the location of the metrics.json, use the command in export_grafana.py to launch the Grafana dashboard
