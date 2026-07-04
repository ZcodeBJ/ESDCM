# Revisiting Multimodal Emotion Recognition in Conversation from the Perspectives of Context and Representation Over-Smoothing



## Requirements
* Python 3.10.13
* PyTorch 1.13.1
* torch_geometric 2.4.0
* torch-scatter 2.1.0
* torch-sparse 0.5.15
* CUDA 11.7

## Preparation

1. Download  [**multimodal-features**](https://www.dropbox.com/scl/fo/veblbniqjrp3iv3fs3z6p/AEzkNgWqPHHzldBZ0zEzr2Y?rlkey=yhlr653c0vnvaf1krpdkla36u&e=1&dl=0) 
2. Save data/iemocap/iemocap_features_roberta.pkl, data/iemocap/IEMOCAP_features.pkl in `data/iemocap/`; Save meld_features_roberta.pkl, data/meld/MELD_features_raw1.pkl in `data/meld/`. 


## Training & Evaluation

1. train ESDCM on IEMOCAP for ERC task
```shell
python code/run_train_erc.py --dataset IEMOCAP --data_dir ../data/iemocap/IEMOCAP_features.pkl \
  --valid_rate 0.0 --modals avl --lr 0.0001 --batch-size 32 --l2 0.0001 --dropout 0.2 --gamma 0.5 --class_weight --reason_flag \
  --mtl --use_clone --hidden_l 400 --hidden_a 400 --hidden_v 400 --persona_l_heads 8 --persona_a_heads 8 --persona_v_heads 8 \
  --persona_l_layer 1 --persona_a_layer 1 --persona_v_layer 1 --interactive_layer 1 --interactive_heads 4 --dropout_forward 0.3 \
  --dropout_persona_lstm_modeling 0.2 --dropout_interactive 0.2 --dropout_persona 0.2 --erc_windows 1 --shift_windows 1 \
  --persona_transform --interactive_windows 1 --epochs 140 --seed 6500 --weight 3
```

2. train ESDCM on MELD for ERC task
```shell
python code/run_train_erc.py --dataset MELD --data_dir ./data/meld/MELD_features_raw1.pkl \
  --valid_rate 0.0 --modals avl --lr 0.0001 --batch-size 32 --l2 0.0001 \
  --mtl --use_clone --hidden_l 200 --hidden_a 200 --hidden_v 200 --persona_transform\
  --persona_l_heads 4 --persona_a_heads 4 --persona_v_heads 4 \
  --persona_l_layer 1 --persona_a_layer 1 --persona_v_layer 1 \
  --interactive_layer 1 --interactive_heads 4 \
  --dropout_forward 0 --dropout_persona_lstm_modeling 0.2 --dropout_interactive 0.2 --dropout_persona 0.2 \
  --erc_windows 1 --shift_windows 1 --interactive_windows 1  --epochs 30 --seed 11407 --weight 5
```


