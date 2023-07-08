# Masked-Language-Model-Fine-Tuning-with-HuggingFace-Transformers

**This model has been deployed on HuggingFace Model Hub and can accessed [here](https://huggingface.co/MUmairAB/bert-based-MaskedLM)**

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on [IMDB Movies Review](https://huggingface.co/datasets/imdb) dataset.
It achieves the following results on the evaluation set:
- Train Loss: 2.4360
- Validation Loss: 2.3284
- Epoch: 20

## Training and validation loss during training

<img src="https://huggingface.co/MUmairAB/bert-based-MaskedLM/resolve/main/Loss%20plot.png" style="height: 432px; width:567px;"/>


## Model description

[DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased)
```
Model: "tf_distil_bert_for_masked_lm"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 distilbert (TFDistilBertMai  multiple                 66362880  
 nLayer)                                                         
                                                                 
 vocab_transform (Dense)     multiple                  590592    
                                                                 
 vocab_layer_norm (LayerNorm  multiple                 1536      
 alization)                                                      
                                                                 
 vocab_projector (TFDistilBe  multiple                 23866170  
 rtLMHead)                                                       
                                                                 
=================================================================
Total params: 66,985,530
Trainable params: 66,985,530
Non-trainable params: 0
_________________________________________________________________
```

## Intended uses & limitations

The model was trained on IMDB movies review dataset. So, it inherits the language biases from the dataset.

## Training and evaluation data

The model was trained on [IMDB Movies Review](https://huggingface.co/datasets/imdb) dataset.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: {'name': 'AdamWeightDecay', 'learning_rate': {'class_name': 'WarmUp', 'config': {'initial_learning_rate': 2e-05, 'decay_schedule_fn': {'class_name': 'PolynomialDecay', 'config': {'initial_learning_rate': 2e-05, 'decay_steps': -60, 'end_learning_rate': 0.0, 'power': 1.0, 'cycle': False, 'name': None}, '__passive_serialization__': True}, 'warmup_steps': 1000, 'power': 1.0, 'name': None}}, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-08, 'amsgrad': False, 'weight_decay_rate': 0.01}
- training_precision: float32

### Training results

| Train Loss | Validation Loss | Epoch |
|:----------:|:---------------:|:-----:|
| 3.0754     | 2.7548          | 0     |
| 2.7969     | 2.6209          | 1     |
| 2.7214     | 2.5588          | 2     |
| 2.6626     | 2.5554          | 3     |
| 2.6466     | 2.4881          | 4     |
| 2.6238     | 2.4775          | 5     |
| 2.5696     | 2.4280          | 6     |
| 2.5504     | 2.3924          | 7     |
| 2.5171     | 2.3725          | 8     |
| 2.5180     | 2.3142          | 9     |
| 2.4443     | 2.2974          | 10    |
| 2.4497     | 2.3317          | 11    |
| 2.4371     | 2.3317          | 12    |
| 2.4377     | 2.3237          | 13    |
| 2.4369     | 2.3338          | 14    |
| 2.4350     | 2.3021          | 15    |
| 2.4267     | 2.3264          | 16    |
| 2.4557     | 2.3280          | 17    |
| 2.4461     | 2.3165          | 18    |
| 2.4360     | 2.3284          | 19    |



### Framework versions

- Transformers 4.30.2
- TensorFlow 2.12.0
- Tokenizers 0.13.3
