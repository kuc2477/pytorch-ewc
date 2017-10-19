# pytorch-ewc
PyTorch implementation of DeepMind's paper [Overcoming Catastrophic Forgetting, PNAS 2017](https://arxiv.org/abs/1612.00796).

![graphic-image](./arts/graphic-image.jpg)

## Results

Continual Learning **without EWC** (*left*) and **with EWC** (*right*).

<img width="300" src="arts/precision-plain.png" /> <img width="300" src="arts/precision-consolidated.png" /> 



## Installation
```
$ git clone https://github.com/kuc2477/pytorch-ewc && cd pytorch-ewc
$ pip install -r requirements.txt
```


## CLI
Implementation CLI is provided by `main.py`


#### Usage
```
$ ./main.py --help
$ usage: EWC PyTorch Implementation [-h] [--hidden-size HIDDEN_SIZE]
                                  [--hidden-layer-num HIDDEN_LAYER_NUM]
                                  [--hidden-dropout-prob HIDDEN_DROPOUT_PROB]
                                  [--input-dropout-prob INPUT_DROPOUT_PROB]
                                  [--task-number TASK_NUMBER]
                                  [--epochs-per-task EPOCHS_PER_TASK]
                                  [--lamda LAMDA] [--lr LR]
                                  [--weight-decay WEIGHT_DECAY]
                                  [--batch-size BATCH_SIZE]
                                  [--test-size TEST_SIZE]
                                  [--fisher-estimation-sample-size FISHER_ESTIMATION_SAMPLE_SIZE]
                                  [--random-seed RANDOM_SEED] [--no-gpus]
                                  [--eval-log-interval EVAL_LOG_INTERVAL]
                                  [--loss-log-interval LOSS_LOG_INTERVAL]
                                  [--consolidate]

optional arguments:
  -h, --help            show this help message and exit
  --hidden-size HIDDEN_SIZE
  --hidden-layer-num HIDDEN_LAYER_NUM
  --hidden-dropout-prob HIDDEN_DROPOUT_PROB
  --input-dropout-prob INPUT_DROPOUT_PROB
  --task-number TASK_NUMBER
  --epochs-per-task EPOCHS_PER_TASK
  --lamda LAMDA
  --lr LR
  --weight-decay WEIGHT_DECAY
  --batch-size BATCH_SIZE
  --test-size TEST_SIZE
  --fisher-estimation-sample-size FISHER_ESTIMATION_SAMPLE_SIZE
  --random-seed RANDOM_SEED
  --no-gpus
  --eval-log-interval EVAL_LOG_INTERVAL
  --loss-log-interval LOSS_LOG_INTERVAL
  --consolidate

```


#### Train
```
$ python -m visdom.server &
$ ./main.py               # Train the network without consolidation.
$ ./main.py --consolidate # Train the network with consolidation.
```


## Reference
- [Overcoming Catastrophic Forgetting, PNAS 2017](https://arxiv.org/abs/1612.00796)

## Author
Ha Junsoo / [@kuc2477](https://github.com/kuc2477) / MIT License
