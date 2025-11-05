from datamodule import WikipediaDataModule
import os
from model import CoordinateAscentModel

  
  

try:
    # Linux: respects CPU affinity masks
    num_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # macOS/Windows: use total logical CPUs
    num_cpus = os.cpu_count() or 1

  
# chenage it later 
checkpoint_path = "wikibio_roberta_roberta/model.ckpt"
# The original path: "/content/unsupervised-deid/wikibio_roberta_roberta/model.ckpt"


model = CoordinateAscentModel.load_from_checkpoint(checkpoint_path)

dm = WikipediaDataModule(

document_model_name_or_path=model.document_model_name_or_path,

profile_model_name_or_path=model.profile_model_name_or_path,

dataset_name='wiki_bio',

dataset_train_split='train[:10%]',

dataset_val_split='val[:20%]',

dataset_version='1.2.0',

num_workers=1,

train_batch_size=64,

eval_batch_size=64,

max_seq_length=128,

sample_spans=False,

)