import torch

# MODEL 
model = "t5"
num_layers = 3  # both encoder and decoder
dropout_rate = 0.1

# TRAINING
num_epochs = 65
batch_size = 16
lr = 1e-4
warmup_steps = 500
num_workers = 4 * torch.cuda.device_count() if torch.cuda.is_available() else 1
save_interval = 659  # num steps per epoch
log_interval = 659

# PATHS
# train_path = "../clean_data/source/train_dataset/"  # for translation
train_path = "../clean_data/source/mlm_training_dataset.txt"  # for mlm/pretraining
validation_path = "../clean_data/source/validation_dataset/"
model_path = "../data/model/t5/"
model_final_path = "../data/model/t5/final/"
logging_path = "../logs/t5/"
load_model_path = None  # path of a trained model to be loaded
load_tokenizer_path = "../clean_data/tokenizer/sp.model"  # path of a trained tokenizer to be loaded else uses t5-small by default