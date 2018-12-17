# SnoringDetection
Frame-Level Models
LstmModel: Processes the features for each frame using a multi-layered LSTM neural net. The final internal state of the LSTM is input to a video-level model for classification. Note that you will need to change the learning rate to 0.001 when using this model.
DbofModel: Projects the features for each frame into a higher dimensional 'clustering' space, pools across frames in that space, and then uses a video-level model to classify the now aggregated features.
FrameLevelLogisticModel: Equivalent to 'LogisticModel', but performs average-pooling on the fly over frame-level features rather than using pre-aggregated features.


source ./venv/bin/activate  # sh, bash, ksh, or zsh


MODELS_DIR=~/yt8m/v2/models
tensorboard --logdir frame:${MODELS_DIR}/frame,video:${MODELS_DIR}/video
