# MNIST-Digit-Classification with PyTorch
 MNIST is the "hello world" of computer vision, so I this project is for me to show off my computer vision skills on a widley known and straightforward dataset. Specifically, I train three different convolutional neural networks (CNNs) of increasing complexity and analyze their performance on the MNIST classification problem

## Data
The dataset is found here https://www.kaggle.com/datasets/hojjatk/mnist-dataset

To read the data I use a prewritten reader found here https://www.kaggle.com/code/hojjatk/read-mnist-dataset

## EDA
The MNIST dataset is well known and is accepted as clean, but for the sake of good practice, I run through a few checks just to make sure everything is in order. First up, I check out class distribution. If any one digit is significantly overrepresented, it may bias models during training. From the plots below, it looks like classes are fairly well distributed. One is slightly overrepresented in training data, but not to the point where I'm worried about it biasing the model towards it. After the plots I have some housekeeping checks to verify the count of training and test images (10,000 and 60,000 respectively) and to make sure that they're all 28x28 greyscale images. Finally, I do one last check just to make sure all of my labels are actually digits 0-9 and there's no sneaky letters or other characters in there. After all checks, it looks like the data is all good so it's time to get things ready for PyTorch.

## Prepping Data for PyTorch
Since I'm using PyTorch, I need to get the data ready for use with PyTorch tensors so I can load them onto my GPU. I do this with very standard `Dataset` and `DataLoader` instances.
- Images reshape from `(28, 28)` to `(1, 28, 28)` for CNNs
- Pixel values normalized from `[0-255]` to `[0-1]`
- Labels cast to `torch.long` type for classification loss (will use `CrossEntropyLoss`)

Data loaders:
- Batch size 64 since image data is pretty small
- Train dataset shuffling each epoch is on to avoid order biasing
- Test dataset shuffling is off to keep it ordered for evaluation

## Models

I make three models for this task. My goal is to evaluate performance on this well understood task using a variety of CNN architectures. I start very basic with two convolution layers and scale up from there. My deeper model will use four convolution layers and my last model will mimic a VGG (Visual Geometry Group) CNN used for generally much higher performance requiring tasks. With the more complicated CNNs, I expect it to be a bit prone to overfitting, so I include batch normalization and dropout.

| Feature | Simple CNN | Deeper CNN | Mini-VGG |
|:---|:---|:---|:---|
| # of Conv layers | 2 | 4 | 7 |
| Conv layer structure | 1 Conv ➔ ReLU ➔ Pool ➔ 1 Conv ➔ ReLU ➔ Pool | 2 Conv ➔ ReLU ➔ 2 Conv ➔ ReLU ➔ Pool (repeat) | (2 Conv ➔ Pool) ➔ (2 Conv ➔ Pool) ➔ (3 Conv ➔ Pool) |
| # of Dense layers | 2 (128, 10 units) | 2 (256, 10 units) | 2 (512, 10 units) |
| Expected overfitting | Low | Medium | High (needs dropout) |
| Suitable for | Quick baseline model | Learning deeper feature hierarchies | Practicing real-world deep model patterns |
| Regularization needed? | No (optional) | Maybe (batchnorm might help) | Yes (batchnorm + dropout to avoid overfitting) |

### Simple CNN Model
- Conv block 1: Conv layer (32 filters, 3x3 kernel) ➔ ReLU ➔ Max Pool (2 stride)
- Conv block 2: Conv layer (64 filters, 3x3 kernel) ➔ ReLU ➔ Max Pool (2 stride)
- Fully connected 1: Dense layer (128 units) ➔ ReLU
- Fully connected 2: Output layer (10 units - one per digit)

### Deeper CNN Model
- Conv Block 1: Conv layer (32 filters, 3x3 kernel) ➔ Conv layer (32 filters, 3x3 kernel) ➔ ReLU ➔ Max Pool (2 stride)
- Conv Block 2: Conv layer (64 filters, 3x3 kernel) ➔ Conv layer (64 filters, 3x3 kernel) ➔ ReLU ➔ Max Pool (2 stride)
- Fully connected 1: Dense layer (256 units) ➔ ReLU
- Fully connected 2:  Output layer (10 units - one per digit)

### Mini VGG Model
- Conv Block 1: Conv layer (64 filters, 3×3 kernel) ➔ ReLU ➔ Conv layer (64 filters, 3×3 kernel) ➔ ReLU ➔ MaxPool2d(2 stride)
- Conv Block 2: Conv layer (128 filters, 3×3 kernel) ➔ ReLU ➔ Conv layer (128 filters, 3×3 kernel) ➔ ReLU ➔ MaxPool2d(2 stride)
- Conv Block 3: Conv layer (256 filters, 3×3 kernel) ➔ ReLU ➔ Conv layer (256 filters, 3×3 kernel) ➔ ReLU ➔ Conv2d(256 filters, 3×3 kernel) ➔ ReLU ➔ MaxPool2d(2 stride)
- Batch normalization after every convolution layer
- Fully Connected 1	Dense (512 units) ➔ ReLU ➔ Dropout(0.5) to hopefully avoid overfitting
- Fully Connected 2	Dense (10 units- again, one per digit)

## Model training
Model training is done with an Adam gradient descent optimizer and cross entropy loss. I use the same training loop for all three models. I also have a learning rate scheduler to allow finer tuning in later epochs. I run each of my models for 30 epochs, but as discussed later that was wholely unnecessary.

## Results
I have a deeper writeup in my notebook, but here's a quick summary of results:
- `MiniVGG` had the best generalization, reaching 99.6% test accuracy with a good distribution of errors without biasing towards any specific misclassifications.
- `SimpleCNN` and `DeeperCNN` actually performed pretty similarly with the simple model performing slightly worse than the medium model with test accuracies of 99.31% and 99.38% respectively. Both models also had a good deal of overlap in their true label-wrong prediction pairs.
- Across all models, misclassified samples were heavily concentrated around visually ambiguous digits (such as 4 vs 9, 5 vs 3), indicating that some errors are likely unavoidable even with human-level inspection.

For future steps, it's hard to say what could be done to increase model performance without significantly more computation. The only approach I would expect to give notably better results would be an ensemble of different complex models, but multiplying needed computation by even 3x or 4x for an extra 0.1% test accuracy is probably not a super reasonably thing to do. Beyond improving performance, training efficiency could probably be improved. As I mentioned earlier, my models all converged well before the 30 epoch limit I gave them. Early stopping would at least halve compute time taken and energy consumption (this was a small scale project, but the environmental impact of machine learning is an important subject).






















