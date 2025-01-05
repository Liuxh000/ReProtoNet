# Refined Prototype Network for Enhanced Few-Shot Art Style Classification
@article{\
  title={Refined Prototype Network for Enhanced Few-Shot Art Style Classification},\
  author={Liu, Xinhang and Liu, Li adn Lu, Yue and Suen, Ching Y.},\
  journal={The Visual Computer},\
  publisher={Springer}\
}


This repository implements a refined prototype network designed for few-shot learning, focusing on art style classification. The project leverages different backbone networks, fine-tuning methods, and transfer learning strategies to improve model performance.

---

## Dataset
The dataset should be organized in the following directory structure:
```
dataset/
    ├── train/
    │     ├── train_class1/
    │     ├── train_class2/
    │     └── ...
    ├── val/
    │     ├── val_class1/
    │     ├── val_class2/
    │     └── ...
```

- **`train/`**: Contains the training data, with each class in its corresponding folder.  
- **`val/`**: Contains the validation data, structured similarly to the training data.

Ensure that the dataset is preprocessed and ready to use before running the training script.

---

## Running the Code

To start training, run:

```bash
python train.py
```

---

## Configuration Options
You can modify the experiment settings by editing the train.py file. The following key options are available:

### 1. Backbone Network
The `backbone` option specifies the feature extraction network.

Example backbones: `resnet34`, `resnet34_mtl`.\
Backbones with the `_mtl` suffix (e.g., resnet34_mtl) use the *Shifting and Scaling (SS)* transfer learning method.\
Note: Models with and without the `_mtl` suffix share the same pre-trained weights (e.g., `resnet34` and `resnet34_mtl` use the same pre-trained weights).
### 2. Fine-tuning Method
The ft option determines the fine-tuning strategy:

`w`: Fine-tune the *network weights*.\
`p`: Fine-tune the *prototypes*.\
`pw`: Alternating fine-tuning of both *weights* and *prototypes*.

---
You can set these options directly in the `train.py` file to configure your experiment.



