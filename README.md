# Implementation of HCGS using PyTorch-Kaldi Speech Recognition Toolkit
PyTorch-Kaldi is an open-source repository for developing state-of-the-art DNN/HMM speech recognition systems. The DNN part is managed by PyTorch, while feature extraction, label computation, and decoding are performed with the Kaldi toolkit. We are using the PyTorch-Kaldi toolkit to implement Hierarchical Coarse-Grain Sparisification of LSTM networks. This GitHub repository is being used solely for the ICML submission and the authors of the paper do not have any association to the authors of PyTorch-Kaldi.

The Pytorch-Kaldi repository can be found [here](https://github.com/mravanelli/pytorch-kaldi)

[See a short introductory video on the PyTorch-Kaldi Toolkit](https://www.youtube.com/watch?v=VDQaf0SS4K0&t=2s)

Changes to the file *neural_network.py* have been made to incorporate HCGS and quantization into training LSTM and MLP networks. Files added to PyTorch-Kaldi are:
* *HCGS.py* - Adds HCGS parameter layer to the network
* *hcgs.py* - Computes the HCGS mask
* *cgs_base.py* - Used by *hcgs.py* to compute HCGS mask
* *binarized_modules.py* - Defines quantized linear layer for in-training quantization
* *save_cgs_mat.py* - Saves HCGS mask and trainied weights after training is complete


## Table of Contents
* [Introduction](#introduction)
* [Prerequisites](#prerequisites)
* [How to install](#how-to-install)
* [Tutorials:](#TIMIT-tutorial)
  * [TIMIT tutorial](#timit-tutorial)
  * [TED-LIUM tutorial](#TED-LIUM-tutorial)
* [Description of the configuration files](#Description-of-the-configuration-files)


## Prerequisites
1. If not already done, install Kaldi (http://kaldi-asr.org/). As suggested during the installation, do not forget to add the path of the Kaldi binaries into $HOME/.bashrc. For instace, make sure that .bashrc contains the following paths:
```
export KALDI_ROOT=/home/mirco/kaldi-trunk
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH
```
As a first test to check the installation, open a bash shell, type "copy-feats" or "hmm-info" and make sure no errors appear.

2. If not already done, install PyTorch (http://pytorch.org/). We tested our codes on PyTorch 0.4.0 and PyTorch 0.4.1. An older version of PyTorch is likely to raise errors. To check your installation, type “python” and, once entered into the console, type “import torch”, and make sure no errors appear.

3. Install kaldi-io package from the kaldi-io-for-python project (https://github.com/vesis84/kaldi-io-for-python). It provides a simple interface between kaldi and python. To install it:

    ```
    git clone https://github.com/vesis84/kaldi-io-for-python.git
    ```

    Remember to add export PYTHONPATH=$PYTHONPATH:<kaldi-io-dir> to $HOME/.bashrc and source it.
    Type python -c "import kaldi_io" to check that the package is correctly installed. You can find more info (including some reading and writing tests) on https://github.com/vesis84/kaldi-io-for-python.


4. We recommend running the code on a GPU machine. Make sure that the CUDA libraries (https://developer.nvidia.com/cuda-downloads) are installed and correctly working. We tested our system on Cuda 9.0, 9.1 and 8.0. Make sure that python is installed (the code is tested with python 2.7). Even though not mandatory, we suggest to use Anaconda (https://anaconda.org/anaconda/python).

## How to install
To install PyTorch-Kaldi, do the following steps:

1. Make sure all the software recommended in the “Prerequisites” sections are installed and are correctly working
2. Clone the PyTorch-Kaldi repository:
```
git clone https://github.com/mravanelli/pytorch-kaldi
```
3.  Go into the project folder and Install the needed packages with:
```
pip install -r requirements.txt
```


## TIMIT tutorial
In the following, we provide a short tutorial of the PyTorch-Kaldi toolkit based on the popular TIMIT dataset.

1. Make sure you have the TIMIT dataset. If not, it can be downloaded from the LDC website (https://catalog.ldc.upenn.edu/LDC93S1).

2. Make sure Kaldi and PyTorch installations are fine. Make also sure that your KALDI paths are corrently working (you should add the kaldi paths into the .bashrc as reported in the section "prerequisites"). For instance, type "copy-feats" and "hmm-info" and make sure no errors appear. 

3. Run the Kaldi s5 baseline of TIMIT. This step is necessary to compute features and labels later used to train the PyTorch neural network. We recommend running the full timit s5 recipe (including the DNN training). This way all the necessary files are created and the user can directly compare the results obtained by Kaldi with that achieved with our toolkit.

4. Compute the alignments (i.e, the phone-state labels) for test and dev data with the following commands (go into $KALDI_ROOT/egs/timit/s5). If you want to use tri3 alignments, type:
```
steps/align_fmllr.sh --nj 4 data/dev data/lang exp/tri3 exp/tri3_ali_dev

steps/align_fmllr.sh --nj 4 data/test data/lang exp/tri3 exp/tri3_ali_test
```

If you want to use dnn alignments (as suggested), type:
```
steps/nnet/align.sh --nj 4 data-fmllr-tri3/dev data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_dev

steps/nnet/align.sh --nj 4 data-fmllr-tri3/test data/lang exp/dnn4_pretrain-dbn_dnn exp/dnn4_pretrain-dbn_dnn_ali_test
```

5. We start this tutorial with a very simple MLP network trained on mfcc features.  Before launching the experiment, take a look at the configuration file  *cfg/TIMIT_baselines/TIMIT_MLP_mfcc_basic.cfg*. See the [Description of the configuration files](#description-of-the-configuration-files-) for a detailed description of all its fields. 

6. Change the config file according to your paths. In particular:
- Set “fea_lst” with the path of your mfcc training list (that should be in $KALDI_ROOT/egs/timit/s5/data/train/feats.scp)
- Add your path (e.g., $KALDI_ROOT/egs/timit/s5/data/train/utt2spk) into “--utt2spk=ark:”
- Add your CMVN transformation e.g.,$KALDI_ROOT/egs/timit/s5/mfcc/cmvn_train.ark
- Add the folder where labels are stored (e.g.,$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali for training and ,$KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev for dev data).

To avoid errors make sure that all the paths in the cfg file exist. **Please, avoid using paths containing bash variables since paths are read literally and are not automatically expanded** (e.g., use /home/mirco/kaldi-trunk/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali instead of $KALDI_ROOT/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali)

7. Run the ASR experiment:
```
python run_exp.py cfg/TIMIT_baselines/ICML_TIMIT.cfg
```

This script starts a full ASR experiment and performs training, validation, forward, and decoding steps.  A progress bar shows the evolution of all the aforementioned phases. The script *run_exp.py* progressively creates the following files in the output directory:

- *res.res*: a file that summarizes training and validation performance across various validation epochs.
- *log.log*: a file that contains possible errors and warnings.
- *conf.cfg*: a copy of the configuration file.
- *model.svg* is a picture that shows the considered model and how the various neural networks are connected. This is really useful to debug models that are more complex than this one (e.g, models based on multiple neural networks).
- The folder *exp_files* contains several files that summarize the evolution of training and validation over the various epochs. For instance, files *.info report chunk-specific information such as the chunk_loss and error and the training time. The *.cfg files are the chunk-specific configuration files (see general architecture for more details), while files *.lst report the list of features used to train each specific chunk.
- At the end of training, a directory called *generated outputs* containing plots of loss and errors during the various training epochs is created.

**Note that you can stop the experiment at any time.** If you run again the script it will automatically start from the last chunk correctly processed. The training could take a couple of hours, depending on the available GPU.

**Debug:** If you run into some errors, we suggest to do the following checks:
1.	Take a look into the standard output.
2.	If it is not helpful, take a look into the log.log file.
3.	If the error is still not clear, run:
```
python run_nn.py exp/TIMIT_MLP_basic/exp_files/train_TIMIT_tr_ep000_ck00.cfg
```
In this way you “manually” run the training of the first chunk of speech an you can see all the errors directly from the standard output.


8. At the end of training, the phone error rate (PER\%) is appended into the res.res file. To see more details on the decoding results, you can go into “decoding_test” in the output folder and take a look to the various files created.  For this specific example, we obtained the following *res.res* file. The trained parameters are also saved in the "exp" folder under the network name in "parameters" as .mat files.



## TED-LIUM tutorial
The steps to run PyTorch-Kaldi on the Librispeech dataset are similar to that reported above for TIMIT. The following tutorial is based on TED-LIUM.

1. Run the Kaldi recipe for timit (at least until # decode using the tri4b model)

2. Compute the fmllr features by running:

```
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)

chunk=train_clean_100
#chunk=dev_clean # Uncomment to process dev
#chunk=test_clean # Uncomment to process test
gmmdir=exp/tri4b

dir=fmllr/$chunk
steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
    --transform-dir $gmmdir/decode_tgsmall_$chunk \
        $dir data/$chunk $gmmdir $dir/log $dir/data || exit 1
        
compute-cmvn-stats --spk2utt=ark:data/$chunk/spk2utt scp:fmllr/$chunk/feats.scp ark:$dir/data/cmvn_speaker.ark
```

3. compute aligmenents using:
```
# aligments on dev_clean and test_clean
steps/align_fmllr.sh --nj 10 data/dev_clean data/lang exp/tri4b exp/tri4b_ali_dev_clean_100
steps/align_fmllr.sh --nj 10 data/test_clean data/lang exp/tri4b exp/tri4b_ali_test_clean_100
```

4. run the experiments with the following command:
```
  python run_exp.py cfg/TEDLIUM/ICML_TEDLIUM.cfg
```


## Description of the configuration files
There are two types of config files (global and chunk-specific cfg files). They are both in *INI* format and are read, processed, and modified with the *configparser* library of python. 
The global file contains several sections, that specify all the main steps of a speech recognition experiments (training, validation, forward, and decoding).
The structure of the config file is described in a prototype file (see for instance *proto/global.proto*) that not only lists all the required sections and fields but also specifies the type of each possible field. For instance, *N_ep=int(1,inf)* means that the fields *N_ep* (i.e, number of training epochs) must be an integer ranging from 1 to inf. Similarly, *lr=float(0,inf)* means that the lr field (i.e., the learning rate) must be a float ranging from 0 to inf. Any attempt to write a config file not compliant with these specifications will raise an error.

Let's now try to open a config file (e.g., *cfg/TIMIT/ICML_TIMIT.cfg*) and let's describe the main sections:

```
[cfg_proto]
cfg_proto = proto/global.proto
cfg_proto_chunk = proto/global_chunk.proto
```
The current version of the config file first specifies the paths of the global and chunk-specific prototype files in the section *[cfg_proto]*.

```
[exp]
cmd = 
run_nn_script = run_nn.py
out_folder = exp/TIMIT_LSTM_fmllr_512c_uni_2l_hcgs_75d32b_75d4b_quant3b
seed = 2233
use_cuda = True
multi_gpu = False
save_gpumem = False
n_epochs_tr = 8
```
The section [exp] contains some important fields, such as the output folder (*out_folder*) and the path of the chunk-specific processing script *run_nn.py*. The field *N_epochs_tr* specifies the selected number of training epochs. Other options about using_cuda, multi_gpu, and save_gpumem can be enabled by the user. The field *cmd* can be used to append a command to run the script on a HPC cluster.

```
[dataset1]
data_name = TIMIT_tr
fea = fea_name=mfcc
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data/train/feats.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data/train/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/mfcc/train_cmvn_speaker.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
	cw_left=0
	cw_right=0
	
	fea_name=fbank
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data/train/feats_fbank.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data/train/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/fbank/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |
	cw_left=0
	cw_right=0
	
	fea_name=fmllr
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/train/feats.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/train/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/train/_fmllr/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |
	cw_left=5
	cw_right=5
lab = lab_name=lab_cd
	lab_folder=/home/dkadetot/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali
	lab_opts=ali-to-pdf
	lab_count_file=auto
	lab_data_folder=/home/dkadetot/kaldi/egs/timit/s5/data/train/
	lab_graph=/home/dkadetot/kaldi/egs/timit/s5/exp/tri3/graph
	
	lab_name=lab_mono
	lab_folder=/home/dkadetot/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali
	lab_opts=ali-to-phones --per-frame=true
	lab_count_file=none
	lab_data_folder=/home/dkadetot/kaldi/egs/timit/s5/data/train/
	lab_graph=/home/dkadetot/kaldi/egs/timit/s5/exp/mono/graph
n_chunks = 5

[dataset2]
data_name = TIMIT_dev
fea = fea_name=mfcc
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data/dev/feats.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data/dev/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/mfcc/dev_cmvn_speaker.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
	cw_left=0
	cw_right=0
	
	fea_name=fbank
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data/dev/feats_fbank.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data/dev/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/fbank/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |
	cw_left=0
	cw_right=0
	
	fea_name=fmllr
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/dev/feats.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/dev/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/dev/_fmllr/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |
	cw_left=5
	cw_right=5
lab = lab_name=lab_cd
	lab_folder=/home/dkadetot/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev
	lab_opts=ali-to-pdf
	lab_count_file=auto
	lab_data_folder=/home/dkadetot/kaldi/egs/timit/s5/data/dev/
	lab_graph=/home/dkadetot/kaldi/egs/timit/s5/exp/tri3/graph
	
	lab_name=lab_mono
	lab_folder=/home/dkadetot/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev
	lab_opts=ali-to-phones --per-frame=true
	lab_count_file=none
	lab_data_folder=/home/dkadetot/kaldi/egs/timit/s5/data/dev/
	lab_graph=/home/dkadetot/kaldi/egs/timit/s5/mono/graph
n_chunks = 1

[dataset3]
data_name = TIMIT_test
fea = fea_name=mfcc
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data/test/feats.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data/test/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/mfcc/test_cmvn_speaker.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
	cw_left=0
	cw_right=0
	
	fea_name=fbank
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data/test/feats_fbank.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data/test/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/fbank/cmvn_test.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |
	cw_left=0
	cw_right=0
	
	fea_name=fmllr
	fea_lst=/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/test/feats.scp
	fea_opts=apply-cmvn --utt2spk=ark:/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/test/utt2spk  ark:/home/dkadetot/kaldi/egs/timit/s5/data-fmllr-tri3/test/_fmllr/cmvn_test.ark ark:- ark:- | add-deltas --delta-order=0 ark:- ark:- |
	cw_left=5
	cw_right=5
lab = lab_name=lab_cd
	lab_folder=/home/dkadetot/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test
	lab_opts=ali-to-pdf
	lab_count_file=auto
	lab_data_folder=/home/dkadetot/kaldi/egs/timit/s5/data/test/
	lab_graph=/home/dkadetot/kaldi/egs/timit/s5/exp/tri3/graph
	
	lab_name=lab_mono
	lab_folder=/home/dkadetot/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_test
	lab_opts=ali-to-phones --per-frame=true
	lab_count_file=none
	lab_data_folder=/home/dkadetot/kaldi/egs/timit/s5/data/test/
	lab_graph=/home/dkadetot/kaldi/egs/timit/s5/exp/mono/graph
n_chunks = 1
```

The config file contains a number of sections (*[dataset1]*, *[dataset2]*, *[dataset3]*,...) that describe all the corpora used for the ASR experiment.  The fields on the *[dataset\*]* section describe all the features and labels considered in the experiment.
The features, for instance, are specified in the field *fea:*, where *fea_name* contains the name given to the feature, *fea_lst* is the list of features (in the scp Kaldi format), *fea_opts* allows users to specify how to process the features (e.g., doing CMVN or adding the derivatives), while *cw_left* and *cw_right* set the characteristics of the context window (i.e., number of left and right frames to append). Note that the current version of the PyTorch-Kaldi toolkit supports the definition of multiple features streams.

Similarly, the *lab* section contains some sub-fields.  For instance, *lab_name* refers to the name given to the label,  while *lab_folder* contains the folder where the alignments generated by the Kaldi recipe are stored.  *lab_opts* allows the user to specify some options on the considered alignments. For example  *lab_opts="ali-to-pdf"* extracts standard context-dependent phone-state labels, while *lab_opts=ali-to-phones --per-frame=true* can be used to extract monophone targets. *lab_count_file* is used to specify the file that contains the counts of the considered phone states. 
These counts are important in the forward phase, where the posterior probabilities computed by the neural network are divided by their priors. PyTorch-Kaldi allows users to both specify an external count file or to automatically retrieve it (using *lab_count_file=auto*). Users can also specify *lab_count_file=none* if the count file is not strictly needed, e.g., when the labels correspond to an output not used to generate the posterior probabilities used in the forward phase, instead, corresponds to the data folder created during the Kaldi data preparation. It contains several files, including the text file eventually used for the computation of the final PER/WER. The last sub-field *lab_graph* is the path of the Kaldi graph used to generate the labels.  

The full dataset is usually large and cannot fit the GPU/RAM memory. It should thus be split into several chunks.  PyTorch-Kaldi automatically splits the dataset into the number of chunks specified in *N_chunks*. The number of chunks might depend on the specific dataset. In general, we suggest processing speech chunks of about 1 or 2 hours (depending on the available memory).

```
[data_use]
train_with = TIMIT_tr
valid_with = TIMIT_dev
forward_with = TIMIT_test
```

This section tells how the data listed into the sections *[datasets\*]* are used within the *run_exp.py* script. 
The first line means that we perform training with the data called *TIMIT_tr*. Note that this dataset name must appear in one of the dataset sections, otherwise the config parser will raise an error. Similarly,  the second and the third lines specify the datasets used for validation and forward phases, respectively.

```
[batches]
batch_size_train = 128
max_seq_length_train = 1000
increase_seq_length_train = False
start_seq_len_train = 100
multply_factor_seq_len_train = 2
batch_size_valid = 128
max_seq_length_valid = 1000
```

*batch_size_train* is used to define the number of training examples in the mini-batch.  The fields *max_seq_length_train* truncates the sentences longer than the specified value. When training recurrent models on very long sentences, out-of-memory issues might arise. With this option, we allow users to mitigate such memory problems by truncating long sentences. Moreover, it is possible to progressively grow the maximum sentence length during training by setting *increase_seq_length_train=True*. If enabled, the training starts with a maximum sentence length specified in start_seq_len_train (e.g, *start_seq_len_train=100*). After each epoch the maximum sentence length is multiplied by the multply_factor_seq_len_train (e.g *multply_factor_seq_len_train=2*).
We have observed that this simple strategy generally improves the system performance since it encourages the model to first focus on short-term dependencies and learn longer-term ones only at a later stage.

Similarly,*batch_size_valid* and *max_seq_length_valid* specify the number of examples in the mini-batches and the maximum length for the dev dataset.

```
[architecture1]
arch_name = LSTM_cudnn_layers
arch_proto = proto/LSTM.proto
arch_library = neural_networks
arch_class = LSTM
arch_pretrain_file = none
arch_freeze = False
arch_seq_model = True
lstm_lay = 512,512
lstm_drop = 0.0,0.0
lstm_quant = True
param_quant = 3,3
lstm_use_laynorm_inp = False
lstm_use_batchnorm_inp = False
lstm_use_laynorm = False,False
lstm_use_batchnorm = True,True
lstm_bidir = False
lstm_act = tanh,tanh
lstm_orthinit = True
arch_lr = 0.0016
arch_halving_factor = 0.5
arch_improvement_threshold = 0.001
arch_opt = rmsprop
opt_momentum = 0.0
opt_alpha = 0.95
opt_eps = 1e-8
opt_centered = False
opt_weight_decay = 0.0
out_folder = 
lstm_hcgs = True
hcgsx_block = 32,4
hcgsx_drop = 75,75
hcgsh_block = 32,4
hcgsh_drop = 75,75
```

The sections *[architecture\*]* are used to specify the architectures of the neural networks involved in the ASR experiments. The field *arch_name* specifies the name of the architecture. The addition of *hcgs* option correspond HCGS block size for the first and second tier and the percentage drop of the connections in first and second tier. The *lstm_quant* and *param_quant* correspond to the quantization options for the network.

Since different neural networks can depend on a different set of hyperparameters, the user has to add the path of a proto file that contains the list of hyperparameters into the field *proto*.  For example,  the prototype file for a standard LSTM model contains the following fields:
```
[proto]
lstm_lay=str_list
lstm_drop=float_list(0.0,1.0)
lstm_use_laynorm_inp=bool
lstm_use_batchnorm_inp=bool
lstm_use_laynorm=bool_list
lstm_use_batchnorm=bool_list
lstm_bidir=bool
lstm_act=str_list
lstm_orthinit=bool
```

Similarly to the other prototype files, each line defines a hyperparameter with the related value type. All the hyperparameters defined in the proto file must appear into the global configuration file under the corresponding *[architecture\*]* section.
The field *arch_library* specifies where the model is coded (e.g. *neural_nets.py*), while *arch_class* indicates the name of the class where the architecture is implemented (e.g. if we set *class=MLP* we will do *from neural_nets.py import MLP*).

The field *arch_pretrain_file* can be used to pre-train the neural network with a previously-trained architecture, while *arch_freeze* can be set to *False* if you want to train the parameters of the architecture during training and should be set to *True* do keep the parameters fixed (i.e., frozen) during training. The section *arch_seq_model* indicates if the architecture is sequential (e.g. RNNs) or non-sequential (e.g., a  feed-forward MLP or CNN). The way PyTorch-Kaldi processes the input batches is different in the two cases. For recurrent neural networks (*arch_seq_model=True*) the sequence of features is not randomized (to preserve the elements of the sequences), while for feedforward models  (*arch_seq_model=False*) we randomize the features (this usually helps to improve the performance). In the case of multiple architectures, sequential processing is used if at least one of the employed architectures is marked as sequential  (*arch_seq_model=True*).

The other hyperparameters are specific of the considered architecture (they depend on how the class MLP is actually implemented by the user) and can define number and typology of hidden layers, batch and layer normalizations, and other parameters.
Other important parameters are related to the optimization of the considered architecture. For instance, *arch_lr* is the learning rate, while *arch_halving_factor* is used to implement learning rate annealing. In particular, when the relative performance improvement on the dev-set between two consecutive epochs is smaller than that specified in the *arch_improvement_threshold* (e.g, arch_improvement_threshold) we multiply the learning rate by the *arch_halving_factor* (e.g.,*arch_halving_factor=0.5*). The field arch_opt specifies the type of optimization algorithm. We currently support SGD, Adam, and Rmsprop. The other parameters are specific to the considered optimization algorithm (see the PyTorch documentation for an exact meaning of all the optimization-specific hyperparameters).
Note that the different architectures defined in *[archictecture\*]* can have different optimization hyperparameters and they can even use a different optimization algorithm.

```
[model]
model_proto = proto/model.proto
model = out_dnn1=compute(MLP_layers1,mfcc)
    loss_final=cost_nll(out_dnn1,lab_cd)
    err_final=cost_err(out_dnn1,lab_cd)
```    

The way all the various features and architectures are combined is specified in this section with a very simple and intuitive meta-language.
The field *model:* describes how features and architectures are connected to generate as output a set of posterior probabilities. The line *out_dnn1=compute(MLP_layers,mfcc)* means "*feed the architecture called MLP_layers1 with the features called mfcc and store the output into the variable out_dnn1*”.
From the neural network output *out_dnn1* the error and the loss functions are computed using the labels called *lab_cd*, that have to be previously defined into the *[datasets\*]* sections. The *err_final* and *loss_final* fields are mandatory subfields that define the final output of the model.

The section forward first defines which is the output to forward (it must be defined into the model section).  if *normalize_posteriors=True*, these posterior are normalized by their priors (using a count file).  If *save_out_file=True*, the posterior file (usually a very big ark file) is stored, while if *save_out_file=False* this file is deleted when not needed anymore.
The *require_decoding* is a boolean that specifies if we need to decode the specified output.  The field *normalize_with_counts_from* set which counts using to normalize the posterior probabilities.

```    
[decoding]
decoding_script_folder = kaldi_decoding_scripts/
decoding_script = decode_dnn.sh
decoding_proto = proto/decoding.proto
min_active = 200
max_active = 7000
max_mem = 50000000
beam = 13.0
latbeam = 8.0
acwt = 0.2
max_arcs = -1
skip_scoring = false
scoring_script = local/score.sh
scoring_opts = "--min-lmwt 1 --max-lmwt 10"
norm_vars = False
```    

The decoding section reports parameters about decoding, i.e. the steps that allows one to pass from a sequence of the context-dependent probabilities provided by the DNN into a sequence of words. The field *decoding_script_folder* specifies the folder where the decoding script is stored. The *decoding script* field is the script used for decoding (e.g., *decode_dnn.sh*) that should be in the *decoding_script_folder* specified before.  The field *decoding_proto* reports all the parameters needed for the considered decoding script. 
