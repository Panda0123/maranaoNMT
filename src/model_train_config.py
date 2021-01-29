import torch

# MODEL 
model = "t5"
num_layers = 3  # both encoder and decoder
dropout_rate = 0.1

# TRAINING
num_epochs = 60
batch_size = 16
lr = 1e-4
warmup_steps = 500
num_workers = 4 * torch.cuda.device_count() if torch.cuda.is_available() else 1
save_interval = 600
log_interval = 600
load_model = False  # wherether to continue training

# PATHS
train_path = "../clean_data/source/train_dataset/"
validation_path = "../clean_data/source/validation_dataset/"
model_path = "../data/model/t5/"
model_final_path = "../data/model/t5/final/"
logging_path = "../logs/t5/"
