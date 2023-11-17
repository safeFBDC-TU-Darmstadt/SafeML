"""
Global configuration script used (or overwritten) in the test and evaluation scripts.
"""

# pick out of {'SecureML', 'Chameleon'}
network = 'SecureML'
batch_size = 1

# pick out of {'malicious', 'semi-honest'}
threat_model = 'malicious'

num_groups = 2
max_byzantine_nodes_per_group = 0
replication_factor = 2 * max_byzantine_nodes_per_group + 1

optimize_communication = True

threads_available = 8

# will only be used in experiments regarding model accuracy (eval_accuracy.py)
epochs = 5

# will only be used in experiments regarding runtime (eval_runtime.py) or communication cost (eval_comm.py)
iterations = 10
train = True

log_warnings = False
