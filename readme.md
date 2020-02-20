# Design Document - Industry Tagging

## Abstract
Developing a pipeline to find the set of industries which any company belongs to, based on the textual description of the company.
 In this pipeline, the Multi-label classification is used where labels are the global set of industry tags. The Deep Learning model used is Recurrent Neural Network followed a Dense Layer and output layer is Sigmoid Layer.


## Data (Feature and Target) Vectorization
The training dataset is a csv(\t separated) file. The (feature, target) column are ('short_description', 'category_list').

The feature is transformed into list of indices (using pretrained Glove vector) on list of words obtained after preprocessing. The list of indices are padded or trimmed at the end to make sure all the instances have uniformity in ssequence lengh(last dimension) after vectorization.

The target column is list of industry which a comapny belongs to. The LabelEncoder is created and stored after reading the dataset. This LabelEncoder is restored and used to transform the set of labels into Multi-Hot Encoder(a company can belong to multiple industries).


## Model Architecture
NN Architecture - 
        (Indices -> Embedding Layer(300-D) : Non-trainable) 
        ----> (RNN -> Hidden: (,100) ) ---> Sigmoid: (, 743) 
        ###Num of dimension in output = 743

Error Function - MultiLabelSoftMarginLoss

Optimization Function - SGD (Stochastic Gradient Descent)



## Training and Plotting Metrics
Each instance is a dict {feature: Tensor, target: Tensor}

Training Dataset Size = ~700k instances
Train: Validation ratio = 90% : 10%

Plot Metrics:  Loss & Validation Accuracy
Plot Frequency: After each batch



## Prediction
Input - Company's Textual Description  

Processing -
        ------> Process to create vector only for feature({'feature': TensorObject})
        ------> Execute trained NN (during training) model and 
        ------> process output (from Multi-hot Encoder to set of Industry Tags)

Output -
        return list of industries in tuple format



## Constants
The hyperparameter and external files used in the entire piepline is mentioned in a global constant JSON file (constants/constants.json) 


