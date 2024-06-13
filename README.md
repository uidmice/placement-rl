# placement-rl

# Description

This repo contains a learning approach called GiPH which learns policies for scheduling tasks to minimise latency accross clusters of devices that are dynamic where computational nodes can move in and out of range. ```main.py``` can be run to create, train, and evaluate a test a model using the GiPH method on generated network and task graphs as well as running it with existing approaches such as Placeto to compare its effectiveness against other baselines.

# paper

https://arxiv.org/abs/2305.14562

To cite this work: 

``` 
@inproceedings{MLSYS2023_3e3eec95,
 author = {Hu, Yi and Zhang, Chaoran and Andert, Edward and Singh, Harshul and Shrivastava, Aviral and Laudon, James and Zhou, Yanqi and Iannucci, Bob and Joe-Wong, Carlee},
 booktitle = {Proceedings of Machine Learning and Systems},
 editor = {D. Song and M. Carbin and T. Chen},
 pages = {164--185},
 publisher = {Curan},
 title = {GiPH: Generalizable Placement Learning for Adaptive Heterogeneous Computing},
 url = {https://proceedings.mlsys.org/paper_files/paper/2023/file/3e3eec95971350490e37a076fdc100ad-Paper-mlsys2023.pdf},
 volume = {5},
 year = {2023}
}
``` 

# Setup

This project is best run in a conda environment to satisfy version requirements, these are contained in requirements.txt and are as follows:


- python 3.8.10
- matplotlib 3.5.1 
- networkx 2.5
- numpy 1.19 
- simpy 4.0.1 
- pytorch 1.13.0
- requests 
- tqdm
- dgl 0.9.1

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

| Command                           | Description                                                                                                                                                                                                                                                                                                                                                              |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```--logdir```                    | allows the user to specify where they want runs of the RL pipeline to be stored, defaults to the runs directory inside the project                                                                                                                                                                                                                                       |
| ```--logdir_suffix```             | each run of the pipeline is stored in a subfolder inside the log directory with a date/time stamp. adding a suffix allows for the user to add a suffix to the date/time stamp. e.g ```python3 main.py``` would create a a folder: 2022-05-27_19-50-41_ inside runs. Whereas ```python3 main.py --logir_suffix suffix``` creates the directory 2022-05-27_19-50-41_suffix |
| ```--disable_cuda```              | allows the user to enable or disable cuda                                                                                                                                                                                                                                                                                                                                |
| ```--noise```                     | noise can take a value of 0-1 which dictates what percentage communication and computation time can vary from the average values of these times. a noise value of 0.2 would allow these times to vary by 20% of the average value                                                                                                                                        |
| ``` --lr ```                      | allows the user to modify the learning rate                                                                                                                                                                                                                                                                                                                              |
| ```--seed ```                     |                                                                                                                                                                                                                                                                                                                                                                          |
| ``` --data_parameters,-p ```      | path to a json text file specifying the training/testing dataset parameters"                                                                                                                                                                                                                                                                                             |
| ``` --train ```                   | training                                                                                                                                                                                                                                                                                                                                                                 |
| ``` --test ```                    | testing (run_folder must be specified)                                                                                                                                                                                                                                                                                                                                   |
| ``` --run_folder ```              | directory to load existing run data                                                                                                                                                                                                                                                                                                                                      |
| ``` --load_train_graphs ```       | path to the customized training graph dataset                                                                                                                                                                                                                                                                                                                            |
| ``` --load_test_graphs ```        | path to the customized testing graph dataset                                                                                                                                                                                                                                                                                                                             |
| ``` --load_train_networks ```     | path to the customized training network dataset                                                                                                                                                                                                                                                                                                                          |
| ``` --load_test_networks ```      | path to the customized testing network dataset                                                                                                                                                                                                                                                                                                                           |
| ``` --embedding_model ```         | file name of the embedding parameters, serialised as a .pk file, model is stored after a set interval of training iterations default to every 5                                                                                                                                                                                                                          |
| ``` --policy_model```             | file name of the policy parameters, serialised as a .pk file, model is stored after a set interval of training iterations default to every 5                                                                                                                                                                                                                             |
| ``` --max_num_training_episodes ```                   | max number of training episodes                                                                                                                                                                                                                                                                                                                                          |
| ``` --min_num_training_episodes ```                   | min number of training episodes                                                                                                                                                                                                                                                                                                                                          |
| ``` --disable_eval```             | disables the evaluation of the model during training on a test data set every eval_frequency episodes                                                                                                                                                                                                                                                                    |
| ``` --num_of_eval_cases ```       | specifies the size of the set of cases to use when evaluating during training                                                                                                                                                                                                                                                                                            |
| ``` --eval_frequency ```          | specifies the frequency at which the model should be evaluated, default is 5                                                                                                                                                                                                                                                                                             |
| ``` --num_testing_cases ```       | size of the test case set, default is 300                                                                                                                                                                                                                                                                                                                                |
| ``` --num_testing_cases_repeat``` | how many times the test cases should be repeated. repetition because the model is stochastic has the potential to output different outcomes                                                                                                                                                                                                                              |

Model/policy parameters

| **Command**    | **Description**                   |
|----------------|-----------------------------------|
| ``` --gamma ```                   | discounting factor                |
| ``` --output_dim ```                   | output dimension of the embedding |
| ``` --hidden_dim ```                   | hidden dimension                  |

Other baselines and variations of GiPH

| **Using other baselines**  | **Description**                                                                                                                   |
|---|-----------------------------------------------------------------------------------------------------------------------------------|
|  ``` --use_placeto ``` | https://arxiv.org/pdf/1906.08879.pdf                                                                                              |
|  ``` --placeto_k``` | specify the number of layers placeto should use                                                                                   |
|  ``` --use_rl_op_est_device ``` | Use RL operator selection and earlist start time device selection https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=993206 |
|  ``` --disable_embedding ``` | Run GiPH without GNN                                                                                                              |
|  ``` --no_edge_features ``` | Remove edge features                                                                                                              |
|  ``` --use_graphsage ``` | Use GraphSAGE for embedding                                                                                                       |

Example Commands:

Train the model on default parameters:

```python main.py --train```

--disable_eval is optional depending on whether you prefer the model being generated faster versus it being evaluated

Load an existing model from a run directory and run a specified amount of tests on it:

``` python main.py --disable_cuda --num_testing_cases 100  --test --run_folder /Users/YourUserName/path/placement-rl/runs/yyyy-mm-dd_hh-mm-ss_(prefix if you specified it)```
