# SafeML: A Privacy-Preserving Byzantine-Robust Framework for Distributed Machine Learning

SafeML is a distributed machine learning framework designed to address privacy and Byzantine robustness concerns throughout the training and inference phases of a machine learning model. It employs secret sharing and data masking techniques to secure all computations, while also utilizing computational redundancy and robust confirmation methods to prevent adverse effects from Byzantine nodes during both model training and inference. The project provides the implementation for the preliminary experimental results, reproducing the accuracy and performance evaluation of SafeML for model training and inference.

## Docker Installation (Recommended)

We have executed our experiments within Docker containers based on the <code>ubuntu:22.04</code> image. For that, we 
provide a <code>Dockerfile</code>. You may simply create the image by executing the commands

```
git clone https://github.com/DataSecPrivacyLab/SafeML.git SafeML
cd SafeML
docker build -t safeml .
```

and run the container by executing the command

```
docker run -it --net=host --privileged --shm-size=10.24gb --name safeml safeml '/bin/bash'
```

You may detach from the interactive shell by pressing <code>Ctrl+p</code>, <code>Ctrl+q</code> and re-attach to the shell 
by executing the command <code>docker attach safeml</code>.

## Manual Installation

You may also install this project manually by running the commands

```
git clone https://github.com/DataSecPrivacyLab/SafeML.git SafeML
cd SafeML
pip install -r SafeML/requirements.txt
```

These commands require an installation of <code>python3</code> and <code>pip</code>. We advise against a manual installation
for LAN / WAN setups if it cannot be guaranteed that the same python version will be used on all servers. 

## Experiment Parameters

In the file <code>configuration.py</code> you can configure the following parameters:

> - **network**: The network to be used in the experiment. The current implementation supports the networks 'SecureML'
    and 'Chameleon'. For each network (and batch size) we configure default learning rates in the file <code>util.constants.py</code>.
> - **batch_size**: The batch size to be used in the experiment.
> - **threat_model**: The threat model to be used in the experiment. Accepted values for this parameter are 'semi-honest'
    and 'malicious'.
> - **num_groups**: The number of groups, i.e., the number of shares created per secret.
> - **max_byzantine_nodes_per_group**: The maximum number of Byzantine nodes assumed in each group. In the 'semi-honest'
    threat model, this will only change the default **replication_factor** (see below). In the 'malicious' threat model, 
    this value will set the number of Byzantine nodes used in the replications of each group. By default, we will use 
    **max_byzantine_nodes_per_group** Byzantine nodes per group and *2*max_byzantine_nodes_per_group+1* replications 
    for each group. You may change this setting by modifying the <code>byzantine_setup</code> parameter in the file 
    <code>util.constants.py</code>.
> - **replication_factor**: The number of replications for each group. By default, this value will be *2\*max_byzantine_nodes_per_group+1*
    (see above).
> - **optimize_communication**: Toggle the optimization (sub-sampling strategy) in the 'semi-honest' threat model. (Currently,
    we do not optimize the collection strategy in the 'malicious' threat model after detecting a Byzantine fault.)
> - **threads_available**: The number of threads to be used for the execution of the experiment. You may set this to twice
    the number of available cores or use it to limit resource utilization.  
> - **epochs**: The number of epochs to be used in experiments regarding the **accuracy** of neural networks in our framework.
    This parameter will be used exclusively in the <code>eval_accuracy.py</code> script.
> - **iterations**: The number of iterations to be used in experiments regarding the **runtime** or **communication cost**
    of neural networks in our framework. This parameter will be used in the scripts <code>eval_runtime.py</code> and 
    <code>eval_comm.py</code>.
> - **train**: Whether to perform training (<code>True</code>) or inference tasks (<code>False</code>) in the experiments
    regarding the **runtime** or **communication cost**.
> - **log_warnings**: Debugging parameter used to enable (<code>True</code>) or disable (<code>False</code>) warnings issued 
    by the Ray framework. Enabling this parameter might give some insight into problems regarding the setup of the servers. 

## Single Machine Setup

On a single machine, you may simply run the experiments by configuring the desired parameters and executing the command

```
python3 ${evaluation_script}
```

(replace <code>${evaluation_script}</code> with the desired evaluation script; the provided evaluation scripts are listed below). We 
evaluate our framework solely on the MNIST data set. In this project, we provide the following evaluation scripts:

> - <code>eval_accuracy.py</code>: A script to evaluate the accuracy of neural networks trained in our framework. By default,
    the accuracy results are evaluated after every 10,000 iterations and printed to the screen. Furthermore, the results 
    are written to a file <code>results/${network}\_epochs\_${epochs}\_bs_${batch_size}.txt</code>.
> - <code>eval_runtime.py</code>: A script to evaluate the runtime of neural networks in our framework. We print the complete
    runtime for all iterations, the runtime of each iteration, the average runtime of all iterations and the average runtime
    of all iterations except the first. The last output is given due to the fact that the first iteration is always significantly
    slower than all other iterations due to the setup time in Ray.
> - <code>eval_comm.py</code>: A script to evaluate the communication cost of neural networks in our framework. Here, we
    print the total number of messages sent and the overall network traffic incurred by the training / inference task(s).

## LAN / WAN Setup

To run our experiments in a LAN or WAN setup, we will create a Ray cluster and deploy our actors over this cluster. For 
that, we will first choose one of the servers as the *head node* of the cluster. On this server we will execute the command

```
ray start --head
```

to limit the number of threads to be used, we may also specify an optional resource parameter. Due to  the limited number 
of available servers, we have used one server for each group of replicated workers. Furthermore, our auxiliary and mediator
nodes are always given by the group *r=0*. To start this group on one server, we may specify the additional resource parameter
<code>--resources='{"cpu": ${threads}, "group0": ${workers_per_group}, "auxiliary": ${workers_per_group}, "mediator": ${workers_per_group}}</code> 
(replace <code>${threads}</code> with the number of threads to use and <code>${workers_per_group}</code>) with the (maximum)
number of workers used per group. You may remove the <code>"cpu"</code> resource if you want to use all available threads on
a server.

To start the other group(s), you may execute the command

```
ray start --address='${ip_head_node}:${assigned_ray_port}'
```

(replace <code>${ip_head_node}</code> with the IP of the head node and <code>${assigned_ray_port}</code> with the port
assigned at the startup of the head node (will be printed with the <code>ray start</code> command)). Again, we will need
to specify the group resources and optionally limit the number of threads with the resource parameter with the resource 
parameter 
<code>--resources='{"cpu": ${threads}, "group${group_num}": ${workers_per_group}}</code> (replace <code${group_num}</code>)
with the group number.

Example: If we have two groups and three replications per group, we may start the first group with

```
ray start --head --resources='{"group0": 3, "auxiliary": 3, "mediator": 3}
```

and the second group with 

```
ray start --address='${ip_head_node}:${assigned_ray_port}' --resources='{"group1": 3}
```

If you want to execute one worker on each server, you may pick one of the servers as the head node and specify the 
<code>group</code>, <code>auxiliary</code> and <code>mediator</code> resource as <code>1</code>.

Finally, you may start any of the evaluation scripts by running the command

```
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit -- python3 ${evaluation_script}
```

on the head node (replace <code>${evaluation_script}</code> with the desired evaluation script).

## Publication ##

Meghdad Mirabi, René Klaus Nikiel, and Carsten Binnig. (2023). **SafeML: A Privacy-Preserving Byzantine-Robust Framework for Distributed Machine Learning Training**. In: The 1st International Workshop on Trustworthy Knowledge Discovery and Data Mining (TrustKDD), Joint Workshop at IEEE International Conference on Data Mining 2023 (ICDM2023), Shanghai, China - 1-4 December, 2023.
<p dir="auto"><a href="https://github.com/safeFBDC-TU-Darmstadt/QFilter/blob/master/SafeML-Paper.pdf
">The paper is available here.</a></p>
