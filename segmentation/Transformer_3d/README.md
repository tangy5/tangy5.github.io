# Model Overview
A transformer-based model for volumetric (3D) multi-organ segmentation task using the BTCV challenge dataset. This model is trained using the UNETR architecture [1].

### Tutorial

A step-by-step tutorial can be found in:

tutorial/unetr_segmentation_3d.ipynb

### Installing Dependencies
Dependencies can be installed using:
``` bash
./requirements.sh
```

### Training command
A typical training command is as follows:
``` bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=1234 __main__.py --amp --amp_scale --batch_size=5 --name=test_run_v1 --conv_block
```

