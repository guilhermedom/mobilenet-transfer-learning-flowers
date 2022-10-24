# MobileNetV2 Transfer Learning Flowers

Transferring [MobileNetV2]'s feature extractor layers to classify [flower images].

---

## Problem Overview

Image classification is a task that has become more and more popular with the growth of [convolutional neural networks] (CNNs). From [handwritten digits], to [datasets with a thousand different classes], better results are constantly achieved in computer vision by CNNs. Nonetheless, CNNs are not one-size-fits-all algorithms. Their applicability should not be understood by only analyzing hyperparameters, but also by taking into consideration if an architecture is really relevant for a given task.

With only 3670 images, for 5 different types of flowers (daisy, dandelion, rose, sunflower and tulip), the [Flowers dataset] is a very small dataset for image classification. The images have a nice quality and can easily go through data augmentation. However, our objective here is to show how to work with small image datasets. Flowers is a great dataset for making proof-of-work models, but its simplicity is also reflected on its label quality. We can easily see that some images are mislabeled and some others do not have actual flowers in them. This constitutes another challenge for this dataset. The image grid below provides an overview on the dataset with random images taken from it:

![mobilenet_flowers_actual_grid](https://user-images.githubusercontent.com/33037020/197432517-d3039a75-3d8e-41d0-99dd-ebb16b2549e2.PNG)

## Analysis Introduction

Many different architectures for CNNs have been proposed already. They have pros and cons, despite being able to outperform classical computer vision algorithms in general. Some of these networks are simple, with only a few thousand of parameters to be adjusted during training. Others have millions of parameters that need to be fed with a big amount of data to be fitted properly. As we are dealing with a small dataset, using a complex model would likely lead to overfitting. Hence, we decided to pick a simple, yet recent, CNN model: [MobileNetV2].

In this repository, we use TensorFlow's Flowers dataset to investigate how the efficient and simple MobileNetV2 can reach optimal results in scenarios with a small amount of data available. With the help of transfer learning, it is possible to use MobileNetV2 without retraining to classify the Flowers dataset. We can load the feature extractor layers from MobileNetV2, and their respective trained weigths on [ImageNet], to directly give a softmax layer the needed input to differentiate the 5 flower types. This is mainly because the last extractor layer of MobileNetV2 acts as an global average pooling layer, what makes the previous convolutional feature maps work as class mappers.

Our model was able to achive 89% accuracy on the Flowers dataset without data preprocessing or augmentation. The [confusion matrix] below highlights our model's difficulties and prowess. Most values reside on the main diagonal, indicating a very good performance with many true positives.

![mobilenet_flowers_cm](https://user-images.githubusercontent.com/33037020/197431678-91465ed9-13bf-450c-90df-3aa3bdd228c1.PNG)

Our model has a bit of trouble trying to distinguish some roses and tulips (indicated in line number 3, column 5, of the confusion matrix). Many of these mistakes may be due to some faulty labeling and class representation in the images. The next image grid shows a couple of random images taken from the dataset. Actual and predicted labels are printed above each image. The only two mistakes presented in the grid refer to the issues pointed out: line 3, column 1, has a daisy flower mislabeled as a rose; line 4, column 1, has a picture without actual flowers, there is only a patch of sunflowers in a towel.

![mobilenet_flowers_grid](https://user-images.githubusercontent.com/33037020/197431688-51374f2d-2a58-41fa-8890-233bf8d63337.PNG)

[//]: #

[MobileNetV2]: <https://arxiv.org/abs/1801.04381>
[flower images]: <https://www.tensorflow.org/datasets/catalog/tf_flowers>
[handwritten digits]: <http://yann.lecun.com/exdb/mnist/>
[datasets with a thousand different classes]: <https://www.image-net.org>
[ImageNet]: <https://www.image-net.org>
[Flowers dataset]: <https://www.tensorflow.org/datasets/catalog/tf_flowers>
[convolutional neural networks]: <https://www.ibm.com/cloud/learn/convolutional-neural-networks>
[confusion matrix]: <https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix>
