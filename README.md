# 프로젝트 : 문장 내 개체간 관계 추출

## Directiory

```
checkpoints
code
├── checkpoints
├── lightning_logs
├── logs
├── prediction
│   └── sample_submission.csv
├── results
├── wandb
├── inference.py
├── load_data.py
├── load_data_marker.py
└── train.py
dataset
├── test
│   └── test_data.csv
└── train
    └── train.csv
README.md
Untitled.ipynb
```

## Usage

1. setting (`config.json` 파일 사용)
    - `config.json` 파일을 열어 파라미터를 확인 후 원하는 값으로 변경
```json
{
    "model_name": "klue/bert-base",
    "batch_size": 16,
    "max_epoch": 20,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "wandb_username": "your_name",
    "wandb_entity": "your_name",
    "wandb_key": "your_key",
    "wandb_project_name": "TEST",
    "use_LSTM": true, 
    "data_path": "../dataset/",
    "warmup_steps": 500, 
    "checkpoint_file": "./checkpoints/last.ckpt" 
}
```

2. Train
- `config.json` 파일 사용 시
```bash
cd code/
python train.py --config config.json
```
- `config.json` 파일 미사용 시
```bash
cd code/
python train.py  --model_name MODEL_NAME --batch_size BATCH_SIZE --max_epoch MAX_EPOCH --learning_rate LEARNING_RATE --weight_decay WEIGHT_DECAY \
--warmup_steps WARMUP_STEPS --log_steps LOG_STEPS --save_steps SAVE_STEPS --shuffle SHUFFLE --use_LSTM USE_LSTM --data_path DATA_PATH \
--wandb_username WANDB_USERNAME --wandb_project_name WANDB_PROJECT_NAME --wandb_entity WANDB_ENTITY --config CONFIG --wandb_key WANDB_KEY
```

3. Inference
- `config.json` 파일에 불러올 `ckpt` 파일을 `"checkpoint_file"` 에 입력한 후
```
python inference.py --config config.json
```