# SnoringDetection
Frame-Level Models
LstmModel: Processes the features for each frame using a multi-layered LSTM neural net. The final internal state of the LSTM is input to a video-level model for classification. Note that you will need to change the learning rate to 0.001 when using this model.
DbofModel: Projects the features for each frame into a higher dimensional 'clustering' space, pools across frames in that space, and then uses a video-level model to classify the now aggregated features.
FrameLevelLogisticModel: Equivalent to 'LogisticModel', but performs average-pooling on the fly over frame-level features rather than using pre-aggregated features.


source ~/python_env/tensorflow/bin/activate  # sh, bash, ksh, or zsh


MODELS_DIR=~/audioset_v1_embeddings/sample_model
tensorboard --logdir frame:${MODELS_DIR}



python train.py --train_data_pattern=${HOME}/audioset_v1_embeddings/bal_train/*.tfrecord --num_epochs=100 --learning_rate_decay_examples=400000 --feature_names=audio_embedding --feature_sizes=128 --frame_features --batch_size=512 --num_classes=527 --train_dir ~/audioset_v1_embeddings/sample_model --model=LstmModel --start_new_model
