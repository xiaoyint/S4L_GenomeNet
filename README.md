# S4L_GenomeNet
### Introduction
One major challenge for machine learning in genomics is the scarcity of labeled data. 
In order to obtain high quality labels, it is often necessary to perform expensive experiments. 
However, in the age of high throughput sequencing, there is a large quantity of unlabeled data that can be used to aid classification through semi-supervised learning. 
Out of the multiple possible avenues to semi-supervised learning, we tackle the problem through representation learning, which has shown promising results in previous studies. 
We investigate this issue using representations for semi-supervised-learning by applying a Contrastive Predictive Coding architecture and a language model predicting the next following nucleotide from a given genome sequence. The representations are evaluated by three downstream tasks from genomics: (1) The differentiation between Gram-positive and Gram-negative bacteria, (2) the differentiation between bacteria, phage-viruses, and other viruses, and (3) the recognition of different chromatin effects of sequence alterations in the human genome.
The performance is compared with fully supervised approaches. 
We find that the language model predicting the next nucleotide from a sequence performs better than the CPC model (concept visualized in Figure below). However, the gain in performance compared to the fully supervised approaches is rather small and further research is needed to draw more definitive conclusions.

<p align="center">
<img src="/images/cpc.png" width=600 align=center>
</p>

## Repository structure
Our Repository consists of the main folder, containing our general functions such as the training script, as well as the scripts for training the self-supervised models. Furthermore, there are five folders:
- The **architectures** folder consists of the functions creating the different encoders, the context networks, as well as the cpc loss function and the layers added on top for supervised training. 
- The **help** folder contains our help-functions. These include functions used inside the training loop, a function creating the AUC for the test set, a script containing multiple functions that simplify our calculations, the environment preparation scripts, and the data preprocessing scripts.
- The **T1_GramStaining** folder contains all scripts for training and evaluation on the Gram Staining Classification Task.
- The **T2_BacteriaVirus** folder contains all scripts for training and evaluation on the Bacteria-Virus Classification Task.
- The **T3_Chromatin** folder contains all scripts for training and evaluation on the Chromatin Features Classification Task.
- The **images** folder contains the plots visualizing the results we obtained.

The three folders containing all scripts for training and evaluation on the different tasks are from the same structure. Each folder includes a ``baseline.R`` script, which runs the training for the baseline model, a ``semi.R`` which runs the supervised model given the self-supervised pretrained model and ``eval.R``, which evaluates the performance on the test data. Additionally, for the Gram Staining and the Bacteria-Virus Classification Task, ``regularization.R`` computes lasso regression given the self-supervised pretrained model and ``tsne.R`` creates T-distributed stochastic neighbor embedding (TSNE) plots of the representations.

## Enviromental setup

Our models are trained using GPUs. 

## Run Self-Supervised Pretraining
To run the Self-Supervised Pretraining, enter the command below. Please note, that here, the relative data_directory is used. If you receive an error message, please check if your current directory is the project repository folder. The first argument denotes the data used for pretraining, the second argument denotes the encoder architecture, the third argument denotes the maximum number of samples per file.

```sh
# CPC ResNet-18, trained on bacterial sequence data with a maximum number of 16 samples per file
Rscript self-supervised_cpc.R bacteria rn18 16

# Language Model, trained on on a combined dataset comprising of bacterial, human, 
# and viral sequence data with a maximum number of 16 samples per file
Rscript self-supervised_lm.R bacteria
```

## Run Supervised Downstream Evaluation
To run the Supervised Downstream Evaluation, enter the command below. Please note, that here, the relative data_directory is used. If you receive an error message, please check if your current directory is the project repository folder. For regularization, the first argument denotes the pretrained_model_folder, the second argument denotes the percentage of labeled data used for training. For training using neural network layers, the first argument denotes the layer added on top, the second argument denotes the pretrained_model_folder, the third argument denotes the percentage of labeled data used for training.

```sh
# By Lasso Regularization on the Gram Staining Classification Target 
# using 1% of the labeled data for training
Rscript T1_GramStaining/regularization.R pretrained_model_folder 1

# By Neural Networks on the Bacteria-Virus Classification Target, 
# with one linear layer added, using 10% of the labeled data for training, finetuning
Rscript T2_BacteriaVirus/semi.R lin TRUE pretrained_model_folder 10

# By Neural Networks on the Chromatin Features Classification Target, 
# with one linear layer added, using 100% of the labeled data for training, linear classification
Rscript T3_Chromatin/semi.R lin FALSE pretrained_model_folder 100
```

## Run Baseline Models for Comparison
To run the Baseline Models for Comparison, enter the command below. Please note, that here, the relative data_directory is used. If you receive an error message, please check if your current directory is the project repository folder. The first argument denotes the encoder architecture, the second argument denotes the layer added on top, the third argument denotes the percentage of labeled data used for training.

```sh
# Baseline Model for the CPC architecture using ResNet-18 as encoder, 
# with one linear layer added for Gram Staining Classification using 1% of the data for training 
Rscript T1_GramStaining/baseline.R rn18 lin 1

# Baseline Model for the CPC architecture using ResNet-50 as encoder, 
# with one linear layer added for Bacteria-Virus Classification using 10% of the data for training 
Rscript T2_BacteriaVirus/baseline.R rn50 lin 10

# Baseline Model for the Language Model, 
# with one linear layer added for Chromatin Features Classification using 100% of the data for training 
Rscript T3_Chromatin/baseline.R lm lin 100
```

## Model Evaluation
For evaluating the model performance using the test data, enter the command below. Please note, that here, the relative data_directory is used. If you receive an error message, please check if your current directory is the project repository folder. The argument after the scriptfile denotes the folder of trained downstream evaluation model.

```sh
# Evaluation of a Gram Staining Classification model
Rscript T1_GramStaining/eval.R downstream_model_folder

# Evaluation of a Bacteria-Virus Classification model
Rscript T2_BacteriaVirus/eval.R downstream_model_folder

# Evaluation of a Chromatin Features Classification model
Rscript T3_Chromatin/eval.R downstream_model_folder
```
### Results

Following performances were achieved:

Gram Staining Classification
<p align="center">
<img src="/images/Gram_all.png" width=600 align=center>
</p>

Bacteria-Virus Classification
<p align="center">
<img src="/images/BacVir_all.png" width=600 align=center>
</p>

Chromatin Features Classification
<p align="center">
<img src="/images/Chromatin_all.png" width=600 align=center>
</p>


## Interpretability of Representations
For the creation of TSNE-plots, enter the command below. Please note, that here, the relative data_directory is used. If you receive an error message, please check if your current directory is the project repository folder. The argument after the scriptfile denotes the folder of pre-trained self-supervised model.

```sh
# TSNE for Gram Staining Classification 
Rscript T1_GramStaining/tsne.R pretrained_model_folder

# TSNE for a Bacteria-Virus Classification
Rscript T2_BacteriaVirus/tsne.R pretrained_model_folder
```

These are the TSNE-plots resulting from self-supervised models trained in our experiments:

Gram Staining Classification
<p align="center">
<img src="/images/tsne_gram.png" width=600 align=center>
</p>

Bacteria-Virus Classification
<p align="center">
<img src="/images/tsne_BacVir.png" width=600 align=center>
</p>




