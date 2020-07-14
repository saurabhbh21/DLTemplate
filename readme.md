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
              
     
 # Project Execution Instruction
 ## Module:
 ```
 **dataset_generate.py** - 
    a) Write the module to Create the training dataset (train.csv)
 
 **dataset.py** - 
    a) Create Tensor from training dataset [set the filename of training dataset (train.csv  in constants/constant.json under ***dataset_config.dataset_file*** key.
    b) Save the LabelEncoder for label to one-hot vector & vice-versa mapping in *data* folder
    
  **model.py** - 
  a) The module responsible for defining Neural Network Architecture, its error and optimization function
        
  **train.py** - 
  a) The module responsible for training the model for action item classification
  b) Once all the dataset and configuration (set the configuration in **constants/constants.json** file) run the following for training the model - ***python module/train.py***
  c) Once the training is over set the best model filename in *constants/constant.json* file under **predict_config.best_model_name** key for prediction with model
  d) Use following command from **project** folder to view training and validation loos & metrics:
          tensorboard --logdir=runs
  
  
  **predict.py**
  a) Pass list on input to predict the output label
          
  ```


 ## Miscellaneous:
 ```
 **utils** Folder - 
    a) **utils.py** - It contains the utility methods related to data-preprocessing, vectorization, finding evaluation metric, etc.
    b) **constant_reader.py** - It is used to read the hyper-parameter and other constants 
    
   **data** Folder
   b) **target.pkl** - Label Encoder for encoding and Decoding Label Name to and from its one-hot representation
    
  
