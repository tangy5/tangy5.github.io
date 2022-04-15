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
python __main__.py --batch_size=1 --opt=adamw --num_steps=45000  --lrdecay --eval_num=100 --name=${NAME} --loss_type=dice_ce --conv_block --res_block --lr=1e-4 --fold=0
```

