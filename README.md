# Interpretable 2-stage representation learning from raw waveform for automatic speech recognition.

A relevance weighting scheme is proposed that allows the interpretation of the speech representations during the forward propagation of the model itself. 
The relevance weighting is achieved using a sub-network approach that performs the task of feature selection. 
A relevance sub-network, applied on the output of first layer of a convolutional neural network model operating on raw speech signals, acts as an acoustic filterbank (FB) layer with relevance weighting. 
A similar relevance sub-network applied on the second convolutional layer performs modulation filterbank learning with relevance weighting. 
The full acoustic model consisting of relevance sub-networks, convolutional layers and feed-forward layers is trained for a speech recognition task.

The script Net_raw_AcFB_Attn_ModFB_Attn_CNN2D_DNN_cuda.py contains the proposed network architecture. It takes the raw speech waveform in batches as input.

