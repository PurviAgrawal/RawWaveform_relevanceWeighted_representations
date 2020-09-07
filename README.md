# 2-stage representation learning from raw waveform for automatic speech recognition.

The consists of an acoustic modeling framework for noise robust speech recognition based on relevance weighting mechanism. The relevance weighting is
achieved using a sub-network approach that performs feature selection. A relevance sub-network is applied on the output of first layer of a convolutional network model operating on raw speech signals while a second relevance sub-network is applied on the second convolutional layer output. The relevance weights for the first layer correspond to an acoustic filterbank selection while the relevance weights in the second layer perform modulation filter selection. The model is trained for a
speech recognition task.

The script Net_raw_AcFB_Attn_ModFB_Attn_CNN2D_DNN_cuda.py contains the proposed network architecture. It takes the raw speech waveform in batches as input, each of size [B,1, 101, 400], for batch size B=32, t=101 raw frames and s=400 samples in each frame.

***************************************************************************************************
Implementation of the paper:

P. Agrawal and S. Ganapathy,"Robust Raw Waveform Speech Recognition Using Relevance Weighted Representations", INTERSPEECH, 2020.

***************************************************************************************************

30-Aug-2020 See the file LICENSE for the licence associated with this software.
