# Unified Training Report

## Run: 20250609_223003
- **Version**: 20250609_223003

#### Data Provenance


##### Train

- **Samples**: 60000
- **Mean**: 0.1307
- **Std**: 0.3081
- **Hash**: 88105458fbeac8a2687f76019d456c284531fcc83fe674f855b44298c2720b08

###### Metadata


####### Statistics

- **Mean**: 0.1307
- **Std**: 0.3081
- **Min**: 0.0000
- **Max**: 1.0000

######## Shape

60000, 28, 28
- **Dtype**: float32

##### Test

- **Samples**: 10000
- **Mean**: 0.1325
- **Std**: 0.3105
- **Hash**: e09cf9414aeaa6cfd4aa68fb9f2be67ed11b32f490bcbc684024609313f3cdf4

###### Metadata


####### Statistics

- **Mean**: 0.1325
- **Std**: 0.3105
- **Min**: 0.0000
- **Max**: 1.0000

######## Shape

10000, 28, 28
- **Dtype**: float32
- **Dataset**: MNIST
- **Timestamp**: 20250609_223003

##### Hashes

- **Data**: 0703b4036f24f810c2deead10c18779612ca88690ff7f1d06ba3b450db407ada

#### Model Provenance


##### Architecture

- **Name**: MNISTModel

###### Layers

|Name|Type|Parameters|
|---|---|---|
|conv1|Conv2d|320|
|conv2|Conv2d|18496|
|dropout1|Dropout|0|
|dropout2|Dropout|0|
|fc1|Linear|1179776|
|fc2|Linear|1290|
- **Total Parameters**: 1199882
- **Timestamp**: 20250609_223003
- **Timestamp**: 20250609_223003

#### Training Provenance

- **Epochs**: 0
- **Batch Size**: 64
- **Learning Rate**: 0.0010

##### Training History

|Epoch|Train Loss|Val Loss|Train Acc|Val Acc|Is Final|Privacy Metrics|
|---|---|---|---|---|---|---|
|1|1.5102|1.5102|0.5072|0.8153|False|{'epsilon': 0.6793925335165759, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|2|1.1734|1.1734|0.7136|0.8657|False|{'epsilon': 0.6931630990809639, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|3|1.3183|1.3183|0.7345|0.8767|False|{'epsilon': 0.7069336646453518, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|4|1.3704|1.3704|0.7458|0.8838|False|{'epsilon': 0.7207042302097397, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|5|1.3964|1.3964|0.7500|0.8911|True|{'epsilon': 0.7344747957741278, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|

##### Final Metrics

- **Loss**: 1.3964
- **Train Accuracy**: 0.7500
- **Test Accuracy**: 0.8911

###### Training History

|Epoch|Train Loss|Val Loss|Train Acc|Val Acc|
|---|---|---|---|---|
|1|1.5102|1.5102|0.5072|0.8153|
|2|1.1734|1.1734|0.7136|0.8657|
|3|1.3183|1.3183|0.7345|0.8767|
|4|1.3704|1.3704|0.7458|0.8838|
|5|1.3964|1.3964|0.7500|0.8911|

###### Privacy Metrics

- **Target Epsilon**: 1.0000
- **Final Epsilon**: 0.7345
- **Delta**: 0.0000

####### Privacy Budget

0.6794, 0.6932, 0.7069, 0.7207, 0.7345
- **Timestamp**: 20250609_223003

#### System Info

- **Python Version**: 3.11.12 (main, Apr  8 2025, 14:15:29) [Clang 17.0.0 (clang-1700.0.13.3)]
- **Torch Version**: 2.3.1

##### Platform

- **System**: Darwin
- **Release**: 24.5.0
- **Machine**: arm64
- **Opacus Version**: 1.1.3
- **Timestamp**: 20250609_223003

#### Hashes

- **Data**: aa1313cb6ef645c64242eee53a23f3394a6b371c855ba58cab20977cd8f65ae9
- **Model**: 7f83e09ac3af96c670a16f99f52077aefcd63456b898bf298b922dce64fcdf31
- **Training**: 5cc4182610b05ef0c519f5e199f837a855382bf66fd7fed3f341c8031e212577
- **Root**: 5c67f1ec0618c591df8379d0cd4b17487c419921e8b33fe941f7cc13158d3988
---

## Run: 20250609_225953
- **Version**: 20250609_225953

#### Data Provenance


##### Train

- **Samples**: 60000
- **Mean**: 0.1307
- **Std**: 0.3081
- **Hash**: 88105458fbeac8a2687f76019d456c284531fcc83fe674f855b44298c2720b08

###### Metadata


####### Statistics

- **Mean**: 0.1307
- **Std**: 0.3081
- **Min**: 0.0000
- **Max**: 1.0000

######## Shape

60000, 28, 28
- **Dtype**: float32

##### Test

- **Samples**: 10000
- **Mean**: 0.1325
- **Std**: 0.3105
- **Hash**: e09cf9414aeaa6cfd4aa68fb9f2be67ed11b32f490bcbc684024609313f3cdf4

###### Metadata


####### Statistics

- **Mean**: 0.1325
- **Std**: 0.3105
- **Min**: 0.0000
- **Max**: 1.0000

######## Shape

10000, 28, 28
- **Dtype**: float32
- **Dataset**: MNIST
- **Timestamp**: 20250609_225953

##### Hashes

- **Data**: 8d9c8600705f8b7c74589c0c3b58d936d567afac6c33ca2c870685f498dd97b3

#### Model Provenance


##### Architecture

- **Name**: MNISTModel

###### Layers

|Name|Type|Parameters|
|---|---|---|
|conv1|Conv2d|320|
|conv2|Conv2d|18496|
|dropout1|Dropout|0|
|dropout2|Dropout|0|
|fc1|Linear|1179776|
|fc2|Linear|1290|
- **Total Parameters**: 1199882
- **Timestamp**: 20250609_225953
- **Timestamp**: 20250609_225953

#### Training Provenance

- **Epochs**: 0
- **Batch Size**: 64
- **Learning Rate**: 0.0010

##### Training History

|Epoch|Train Loss|Val Loss|Train Acc|Val Acc|Is Final|Privacy Metrics|
|---|---|---|---|---|---|---|
|1|1.4565|1.4565|0.5388|0.8156|False|{'epsilon': 0.6793925335165759, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|2|1.2211|1.2211|0.7060|0.8571|False|{'epsilon': 0.6931630990809639, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|3|1.3320|1.3320|0.7280|0.8757|False|{'epsilon': 0.7069336646453518, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|4|1.4142|1.4142|0.7330|0.8795|False|{'epsilon': 0.7207042302097397, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|5|1.4243|1.4243|0.7443|0.8862|True|{'epsilon': 0.7344747957741278, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|

##### Final Metrics

- **Loss**: 1.4243
- **Train Accuracy**: 0.7443
- **Test Accuracy**: 0.8862

###### Training History

|Epoch|Train Loss|Val Loss|Train Acc|Val Acc|
|---|---|---|---|---|
|1|1.4565|1.4565|0.5388|0.8156|
|2|1.2211|1.2211|0.7060|0.8571|
|3|1.3320|1.3320|0.7280|0.8757|
|4|1.4142|1.4142|0.7330|0.8795|
|5|1.4243|1.4243|0.7443|0.8862|

###### Privacy Metrics

- **Target Epsilon**: 1.0000
- **Final Epsilon**: 0.7345
- **Delta**: 0.0000

####### Privacy Budget

0.6794, 0.6932, 0.7069, 0.7207, 0.7345
- **Timestamp**: 20250609_225953

#### System Info

- **Python Version**: 3.11.12 (main, Apr  8 2025, 14:15:29) [Clang 17.0.0 (clang-1700.0.13.3)]
- **Torch Version**: 2.3.1

##### Platform

- **System**: Darwin
- **Release**: 24.5.0
- **Machine**: arm64
- **Opacus Version**: 1.1.3
- **Timestamp**: 20250609_225953

#### Hashes

- **Data**: 091fb6f271202846bd4bdb22a6454048861680ec57a9e7fffc92b61304c35b3f
- **Model**: 8c53ab87bf8398689d856a1c4c63abdeda9c6a154b2c5ba97343a30129c86600
- **Training**: 835084a2d13c635991c963bca67f9eae428affe7997755d88c1830f4e8d3fbad
- **Root**: b21d6ddc22f4532a0d45d1b5a23b5d8ee42172f51d8a634a465ae4d1de565332
---

## Run: 20250609_193101
- **Version**: 20250609_193101

#### Data Provenance


##### Train

- **Samples**: 60000
- **Mean**: 0.1307
- **Std**: 0.3081
- **Hash**: 88105458fbeac8a2687f76019d456c284531fcc83fe674f855b44298c2720b08

###### Metadata


####### Statistics

- **Mean**: 0.1307
- **Std**: 0.3081
- **Min**: 0.0000
- **Max**: 1.0000

######## Shape

60000, 28, 28
- **Dtype**: float32

##### Test

- **Samples**: 10000
- **Mean**: 0.1325
- **Std**: 0.3105
- **Hash**: e09cf9414aeaa6cfd4aa68fb9f2be67ed11b32f490bcbc684024609313f3cdf4

###### Metadata


####### Statistics

- **Mean**: 0.1325
- **Std**: 0.3105
- **Min**: 0.0000
- **Max**: 1.0000

######## Shape

10000, 28, 28
- **Dtype**: float32
- **Dataset**: MNIST
- **Timestamp**: 20250609_193101

##### Hashes

- **Data**: 18afc290e68c30b11411b7ee72dde4ce520fb775006e5119f600c8537c6a00eb

#### Model Provenance


##### Architecture

- **Name**: MNISTModel

###### Layers

|Name|Type|Parameters|
|---|---|---|
|flatten|Flatten|0|
|fc1|Linear|100480|
|relu|ReLU|0|
|dropout|Dropout|0|
|fc2|Linear|1290|
- **Total Parameters**: 101770
- **Timestamp**: 20250609_193101
- **Timestamp**: 20250609_193101

#### Training Provenance

- **Epochs**: 0
- **Batch Size**: 64
- **Learning Rate**: 0.0010

##### Training History

|Epoch|Train Loss|Val Loss|Train Acc|Val Acc|Is Final|Privacy Metrics|
|---|---|---|---|---|---|---|
|1|1.2266|1.2266|0.6514|0.7905|False|{'epsilon': 0.6793925335165759, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|2|0.6037|0.6037|0.8051|0.8618|False|{'epsilon': 0.6931630990809639, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|3|0.5318|0.5318|0.8410|0.8765|False|{'epsilon': 0.7069336646453518, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|4|0.5206|0.5206|0.8533|0.8868|False|{'epsilon': 0.7207042302097397, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|
|5|0.5151|0.5151|0.8622|0.8914|True|{'epsilon': 0.7344747957741278, 'delta': 1e-05, 'noise_multiplier': 1.0, 'privacy_budget': [0.6793925335165759, 0.6931630990809639, 0.7069336646453518, 0.7207042302097397, 0.7344747957741278]}|

##### Final Metrics

- **Loss**: 0.5151
- **Train Accuracy**: 0.8622
- **Test Accuracy**: 0.8914

###### Training History

|Epoch|Train Loss|Val Loss|Train Acc|Val Acc|
|---|---|---|---|---|
|1|1.2266|1.2266|0.6514|0.7905|
|2|0.6037|0.6037|0.8051|0.8618|
|3|0.5318|0.5318|0.8410|0.8765|
|4|0.5206|0.5206|0.8533|0.8868|
|5|0.5151|0.5151|0.8622|0.8914|

###### Privacy Metrics

- **Target Epsilon**: 1.0000
- **Final Epsilon**: 0.7345
- **Delta**: 0.0000

####### Privacy Budget

0.6794, 0.6932, 0.7069, 0.7207, 0.7345
- **Timestamp**: 20250609_193101

#### System Info

- **Python Version**: 3.11.12 (main, Apr  8 2025, 14:15:29) [Clang 17.0.0 (clang-1700.0.13.3)]
- **Torch Version**: 2.7.1

##### Platform

- **System**: Darwin
- **Release**: 24.5.0
- **Machine**: arm64
- **Opacus Version**: 1.1.3
- **Timestamp**: 20250609_193101

#### Hashes

- **Data**: ce5df309d6271fbf67d27fee0ecf6b71ccd5509d2bc6b29099a16e2d41ec23f6
- **Model**: 8ef7e7352d19eb610cdf38e91de717c050468fc7b7a5245d25ab47812895057c
- **Training**: ab92e789590bf03a73119d81070c60cd8af47d7cbb5e3294618a4e8a20c126d9
- **Root**: 5946ed33aaeb7428384a98b386b691d66143e7e74fe6ee6a5ce00dcb5ef9334f
---
