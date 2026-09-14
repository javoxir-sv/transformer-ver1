from pathlib import Path
def get_config():
    return {
            "batch_size" : 26, # ~14.1 GB memory usage | sweet spot for google colab free tier T4 gpus 
            "num_epochs" : 10,
            "lr" : 1e-4,
            "seq_len" : 350,
            "d_model" : 512,
            "src_lang" : "en",
            "tgt_lang" : "it",
            "model_dir" : "weigths",
            "model_prefix" : "tmodel_",
            "preload" : None, # e.g.: "07"
            "tokenizer_file" : "tokenizer_{0}.json",
            "experiment_name" : "runs/tmodel",
            }

def get_weights_filepath(config, epoch:str):
    model_dir = config['model_dir']
    model_prefix = config['model_prefix']
    model_filename = f"{model_prefix}{epoch}.pt"
    return str(Path('.')/model_dir/model_filename)

