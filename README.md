# README
***
I made some change based on the repository below, and made these models support python 3.6 and pytorch 1.2.

These models are used as benchmarks in my paper

Some key changes:

1. Only train the model on train data and only evaluate it on test data

    The previous implementation evaluated the model based on all data (train+test), which is hardly right in any situation.

2. Make the model a pure imputation model

    Set all lose weight to imputation loss=1, classification loss=0

3. Support Python 3.6

4. Limited by the model's design (input dim == output dim), in order to impute only one variable in the TS data, I can only input TS with 1 dim. 

5. There are two branches in this repository: newmodel and old model
    * Newmodel: based on imputation here: https://github.com/caow13/BRITS
    * Oldmodel: based on imputation here: https://github.com/NIPS-BRITS/BRITS/tree/master/models

6. To use these model, need to generate dataset with propoer structure first
    Will put this piece of code later.
***

It is a pytorch implemention of paper "BRITS: Bidirectional Recurrent Imputation for Time Series, Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li Yitan Li. (NerIPS 2018)". The paper can be found here. http://papers.nips.cc/paper/7911-brits-bidirectional-recurrent-imputation-for-time-series

To train the BRIST model, first please unzip the PhysioNet data into ***raw*** folder, including the label file ***Outcomes-a.txt***.

To run the model:
* make a empty folder named ***json***, and run inpute_process.py.
* run different models:
    * e.g., RITS_I: python main.py --model rits_i --epochs 1000 --batch_size 64 --impute_weight 0.3 --label_weight 1.0 --hid_size 108
    * for most cases, using impute_weight=0.3 and label_weight=1.0 lead to a good performance. Also adjust hid_size to control the number of parameters

