# placement-rl

# Description

This repo contains a learning approach called GiPH which learns policies for scheduling tasks to minimise latency accross clusters of devices that are dynamic where computational nodes can move in and out of range. ```main.py``` can be run to create, train, and evaluate a test a model using the GiPH method on generated network and task graphs as well as running it with existing approaches such as Placeto to compare its effectiveness against other baselines.

# paper

link

# Setup

This project is best run in a conda environment to satisfy version requirements, these are contained in requirements.txt and are as follows:


- python 3.8.10
- matplotlib 3.5.1 
- networkx 2.5
- numpy 1.19 
- simpy 4.0.1 
- pytorch 
- requests 
- tqdm
- dgl

Conda command:

``` 
conda create -n placement matplotlib=3.5.1 networkx=2.5 numpy=1.19 python=3.8.10 simpy=4.0.1 pytorch requests tqdm dgl  
```

System requirements:

if running on an arm mac you need to run the x86 version of conda through rosetta as DGL requires x86 architecture

# Operation

the pipeline can be run to create a trained model and testing results by running main.py in the conda environment

```main.py``` supports command line arguements to modify the operation and paramters of the pipeline:

Default parameters can be found within main.py and are selected when the user does not input that parameter as an arugement when calling main.py

| Command  |  Description |
|---|---|
| ```--logdir```  | allows the user to specify where they want runs of the RL pipeline to be stored, defaults to the runs directory inside the project  |
| ```--logdir_suffix```  | each run of the pipeline is stored in a subfolder inside the log directory with a date/time stamp. adding a suffix allows for the user to add a suffix to the date/time stamp. e.g ```python3 main.py``` would create a a folder: 2022-05-27_19-50-41_ inside runs. Whereas ```python3 main.py --logir_suffix suffix``` creates the directory 2022-05-27_19-50-41_suffix  |
| ```--disable_cuda```  | allows the user to enable or disable cuda   |
| ```--noise```  | noise can take a value of 0-1 which dictates what percentage communication and computation time can vary from the average values of these times. a noise value of 0.2 would allow these times to vary by 20% of the average value |
| ``` --lr ```  | allows the user to modify the learning rate  |
|  ```--seed ``` | seed to be used for randomly generating the graphs that the pipeline solves  |
| ``` --device_net_path```  |  input directory for device network parameters"  |
|  ``` --op_net_path ``` |  input directory for operator network params |
| ``` --data_parameters ```  |  path to a json text file specifying the training/testing dataset parameters" |
|  ``` --disable_train ``` | disables training, use if you want to test an existing trained model  |
| ``` --disable_test ```  | disables testing of the trained model  |
|  ``` --run_folder ``` | directory to load existing run data  |
| ``` --embedding_model ```  | file name of the embedding parameters, serialised as a .pk file, model is stored after a set interval of training iterations default to every 5|
|  ``` --policy_model``` | file name of the policy parameters, serialised as a .pk file, model is stored after a set interval of training iterations default to every 5  |
|  ``` --test_parameters ``` | json text file specifying the testing dataset parameters |
|  ``` --disable_eval``` | disables the evaluation of the model during training on a test data set every eval_frequency episodes  |
|  ``` --num_of_eval_cases ``` | specifies the size of the set of cases to use when evaluating during training  |
|  ``` --eval_frequency ``` | specifies the frequency at which the model should be evaluated, default is 5  |
|  ``` --num_testing_cases ``` | size of the test case set, default is 200  |
|  ``` --num_testing_cases_repeat``` | how many times the test cases should be repeated. repetition because the model is stochastic has the potential to output different outcomes  |

To compare this pipeline against other existing ones the pipeline can be run with other existing models to compare it against GiPh

| **Using other baselines**  |  **Description** |
|---|---|
|  ``` --use_placeto ``` | https://arxiv.org/pdf/1906.08879.pdf  |
|  ``` --placeto_k``` | specify the number of layers placeto should use  |
|  ``` --use_edgnn ``` |   |
|  ``` --use_radial_mp ``` |   |
|  ``` --radial_k ``` | Number of layers for radial (default: 8)'  |
|  ``` --use_rl_op_est_device ``` |  Use RL operator selection and earlist start time device selection https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=993206 |


Example Commands:

Train the model on default parameters:

```python3 main.py --disable_cuda --disable_test --disable_eval```

--disable_eval is optional depending on whether you prefer the model being generated faster versus it being evaluated

Load an existing model from a run directory:

``` ```
