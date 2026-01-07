# Refined Prototype Network for Enhanced Few-Shot Art Style Classification
@article{\
  title={Refined Prototype Network for Enhanced Few-Shot Art Style Classification},\
  author={Li Liu , Huiqin Yu , Lei Wang , Yuexuan Tang , Xinhang Liu a,∗ , Ching Y. Suen},\
  journal={Expert system with applications},\
  publisher={Springer}\
}


This repository implements a refined prototype network designed for few-shot learning, focusing on art style classification. The project leverages different backbone networks, fine-tuning methods, and transfer learning strategies to improve model performance.

---

## Requirements
Python = 3.7\
PyTorch = 1.8.0\
torchvision = 0.9.0\
CUDA = 11.1

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

### Dataset Download and Divide
`WikiArt`：The dataset can download at  [https://archive.org/details/wikiart-dataset](https://archive.org/details/wikiart-dataset). we designate the 18 classes with the most samples as the training set and the remaining 9 classes as the test set.

`MultitaskPainting100k`：The dataset can download at [http://www.ivl.disco.unimib.it/activities/paintings](http://www.ivl.disco.unimib.it/activities/paintings/). we exclude categories with fewer than 20 samples and use 70 of the remaining 110 categories as the training set, reserving 40 categories for the test set.

`Painting-91`：The dataset can download at [https://archive.org/details/wikiart-dataset](http://www.cat.uab.cat/~joost/painting91.html). we select the 8 classes with the largest number of samples for training and use the remaining 5 classes for testing.
`Architectural Styles`：The dataset can download at [https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset],al., 2014) contains 10,113 images of 25 architectural styles. Since architectural images exhibit visual character- istics (geometry, symmetry, material textures) fundamentally different from paintings, we use this dataset exclusively as a cross-domain test set. It serves to evaluate how well prototype refinement learned from art images transfers to a distinct structural domain.
`International Architectural Styles Combined`：The dataset can download at [https://www.kaggle.com/datasets/jungseolin/international-architectural-styles-combined], includes 14,833 images covering 45 building styles. Like the previous dataset, it is used entirely as a \emph{cross-domain test set}. 
Its broader coverage of global architectural traditions further challenges the model’s ability to generalize refined prototypes beyond the art domain.

Ensure that the dataset is preprocessed and ready to use before running the training script.

---

## Running the Code

To start training, run:

```bash
python train.py
```

To start testing, run:

```bash
python test.py
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





