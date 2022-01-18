# MNIST shuffle
### Introduction
The digits in the test set of the 
[MNIST dataset](http://yann.lecun.com/exdb/mnist/)
 got cut in half vertically and shuffled
around. The aim of this task is to make an algorithm to pair together the matching tops and bottoms 
from the shuffled the images in MNIST test set. 

### Requirements
The required packages can be found in *config/env_files/mnist_shuffle_env.yml*. 
Dependencies could be installed by running:
> conda env create -f config/env_files/mnist_shuffle_env.yml

### Configuration
The experiments are run according to configurations. The config files for those can be found in 
*config/config_files*.
Configurations can be based on each other. This way the code will use the parameters of the specified 
base config and only the newly specified parameters will be overwritten.
 
The base config file is *siamese.yaml*. A hpo example can be found in *siamese_hpo.yaml*
which is based on *siamese.yaml* and does hyperparameter optimization only on the specified parameters.

### Arguments
The code should be run with arguments: 

--id_tag specifies the name under the config where the results will be saved \
--config specifies the config name to use (eg. config "base" for *config/config_files/base.yaml*)\
--mode can be 'train', 'val', 'test' or 'hpo' 

### Required data
The required data fules should bu undder the specified dataset path like:
> data: \
  &emsp; params: \
  &emsp;&emsp; dataset_path: 'dataset' 

During first trainor val the code will automatically generate the train-val set 
from the oiginal train set. \
During test the images are loaded from the test set .  

### Saving and loading experiment
The save folder for the experiment outputs can be set in the config file like:
> id: "siamese"\
  env: \
  &emsp; result_dir: 'results'

All the experiment will be saved under the given results dir: {result_dir}/{config_id}/{id_tag arg}
1. config file
2. best scores
3. the best model and current parameters for continuing training
4. the plain best models

If the result dir already exists and contains a model file then the experiment will automatically resume
(either resume the training or use the trained model for inference.)

### Usage
##### Training
To train the model use:
> python run.py --config siamese --mode train

#### Eval
For eval the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar*. During eval the validation images will be inferenced and the accuracy will be calculated.
> python run.py --config siamese --mode val

#### Test
For test the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar*. During test the accuracy will be calculated and the original, shuffled and unshuffled predictions 
will be saved in image files.\
A pretrained model can be found in [here](https://drive.google.com/drive/folders/13cb6XyZ-Zx6kervrjHtlkcA9QNZIOCyz?usp=sharing). 
For simplicity it is recommended to copy it under *results/siamese/base* 
and just change the dataset path to yours in *config/config_files/siamese.yaml*.
> python run.py --config siamese --mode test

#### HPO
For hpo use:
> python run.py --config siamese_hpo --mode hpo

### Example of the results:
The best unshuffleing accuracy in the val set is score is: **55%**, and on the test set **54%**.
