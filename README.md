# PLMCCL-TP
PLMCCL-TP: The Protein Language Model and Clustering Method Based on Contrastive Learning Applied to the Multifunctional Therapeutic Peptide Identification Model

## Introduction

In this paper, we developed a model named PLMCCL - TP for predicting multifunctional therapeutic peptides (MFPTs). Compared with existing methods, this work has the following advantages:
(1)Make full use of the protein language model to extract protein sequence information.
(2)Utilize contrastive learning to enhance the distinguishability among different peptides.
(3)Employ the Gaussian Mixture Model (GMM) clustering method to better extract the information of related peptides.
(4)Use multi-scale convolution to extract sequence information from different distances.


The framework of the PLMCCL-TP method for MFTP prediction is described as follows:
![draft](./figures/PLMCCL-TP.jpg)

# PLMCCL-TP
| FILE NAME            | DESCRIPTION                                                                                  |
|:---------------------|:---------------------------------------------------------------------------------------------|
| PLMCCL.py            | the main file of PLMCCL-TP predictor (include data reading, encoding, and data partitioning) |
| contrastive_laean.py | Contrastive learning to fine-tune protein language models                                    |
| model.py             | train model                                                                                  |
| contrastive_utils.py | utils used to builad Contrastive learning                                                            |
| util.py              | utils used to build models                                                                   |
| loss_functions.py    | loss functions used to train models                                                          |
| evaluation.py        | evaluation metrics (for evaluating prediction results)                                       |
| dataset              | data                                                                                         |                                          |                                    |                                                                        |
## Training and test PLMCCL-TP model
**1.** Clone this repository by `git clone git@github.com:jiangkun-lin/PLMCCL-TP.git`.

**2.** Install the protein language models. The [ESMFold](https://github.com/facebookresearch/esm) and [ProtTrans](https://github.com/agemagician/ProtTrans) can be installed, followed by their official tutorials. The pre-trained ProtT5-XL-UniRef50 model can be downloaded [here](https://zenodo.org/record/4644188) .   

**3.** Install the requirements

**4.** Contrastive learning to fine-tune protein language models(contrastive_laean.py)

**5.** train the model(PLMCCL.py )

## Contact
Please feel free to contact us if you need any help.

