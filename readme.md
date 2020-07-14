# Project Setup Instruction
1) Setup and Activate virtual environment (name of environment - **env**)
      - Ubuntu: https://linuxize.com/post/how-to-create-python-virtual-environments-on-ubuntu-18-04/

2) Install requirements and Execute script files
    ```
    a) pip install -r requirements.txt
    b) sh scripts/script.sh
    c) sh scripts/glove.sh
    ```

3) Execute project setup File
    ```
    python setup.py develop
    ```
    
4) Save the Dataset in **dataset** folder as follows:
     - *email.csv* - From Enron Email Dataset in Kaggle competition
     - *actions.csv* - File containing email sentences which are actionable
     
     
 ## Project Pipeline 
1) Objective1: Refer (objective1/Objective1 - Documentation.txt)

2) Objective2: Refer (objective2/Objective2 - Documentation.txt)

3) Run following command to view result of trained model described in Objective2 (objective2/Objective2 - Documentation.txt)
    ```
    tensorboard --logdir=runs
    ```
    
     
     
 # Project Execution Instruction
 ## Objective1:
 ```
 **heuristic.py** - 
      a) The core functional module for classifying a sentences based on heuristic model into actionable & non-actionable.
      
 **dataset.py** - 
      a) The module to create dataset wih is_actionable for each sentence of emails in email.csv file.
      b) The dataset is saved as *sentences_actionable_heuristic.csv*
      
 **api.py** - 
      a) The module which takes raw email and returns list of pairs for (sentence, is_actionable).
      b) Run the *api.py* in *objective1* Folder and test with *main_heuristic.py* file in *project* folder.
 ```
 
 ## Objective2:
 ```
 **dataset_generate.py** - 
    a) Create the training dataset with examples in *actions.csv* as positive and negative sentences from heuristic for emails in *emails.csv* as negative
 
 **dataset.py** - 
    a) Create Tensor from training dataset [set the filename of training dataset (train.csv - geneted via dataset_generate.py/ sentences_actionable_heuristic.csv - generated via objective1/dataset.py) in constants/constant.json under ***objective2.dataset_config.dataset_file*** key.
    b) Save the LabelEncoder for label to one-hot vector & vice-versa mapping in *data* folder
    
  **train.py** - 
  a) The module responsible for defining Neural Network Architecture, its error and optimization function
        
  **train.py** - 
  a) The module responsible for training the model for action item classification
  b) Once all the dataset and configuration (set the configuration in **constants/constants.json** file) run the following for training the model - ***python objective2/train.py***
  c) Once the training is over set the best model filename in *constants/constant.json* file under **objective2.predict_config.best_model_name** key for prediction with model
  
  
  **predict.py**
  a) Pass list on sentences to predict the is_actionable (actionable/non-actionable) for each sentence
  
  
  **api.py** 
  a) An api for predicting using trained model for sentences in a raw email.
  b) b) Run the *api.py* in *objective1* Folder and test with *main_predict.py* file in *project* folder for predicting is_actionable for sentences in a raw email.
        
  ```


 ## Miscellaneous:
 ```
 **utils** Folder - 
    a) **utils.py** - It contains the utility methods related to data-preprocessing, vectorization, finding evaluation metric, etc.
    b) **constant_reader.py** - It is used to read the hyper-parameter and other constants in both *objective1* & *objective2*
    
   **data** Folder
   a) **pronoun.json** - It is borrowed from *spacy* GitHub code. It has list of all pronouns (with properties) for use in heuristic model.
   b) **target.pkl** - Label Encoder for encoding and Decoding Label Name to and from its one-hot representation
    
  
