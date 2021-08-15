# Pneumonia detection

## Summary
This project explores the application of machine learning to assist experts in the diagnosis and analysis of thorax diseases by prototyping a deep learning algorithm to detect visual signals of pneumonia infection. A collection of Convolutional Neural Networks (CNN)s which include VGG16 (Simonyan & Zisserman 2014), ResNet50V2 (He et al. 2016b), NasNetMobile (Zoph & Le 2016) and InceptionResNetV2 (Szegedy et al. 2017) were modified, trained and tested on the Mendeley chest x-ray dataset (Kermany et al. 2018) for the task. The F1-score achieved by the best model in this project (VGG16 model 1 version 1.1.1.0.0) shows that it outperformed similar works reviewed such as Rajpurkar et al. (2017); 0.942 vs 0.435 respectively.


## Resources

Hardware: Google Collab GPU

Software: Tensorflow, Keras

Dataset: 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

https://data.mendeley.com/datasets/rscbjbr9sj/2


## Acknowledgement

The models initial build from the guidance from community and resources such as Tensorflow Community, Google Colab Community, Medium and other resources. Specific projects which are closely related to this project can be found below.

- Google Collab - rock, paper, scissors notebook: <br>
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb#scrollTo=LWTisYLQM1aM

- Easy to understand notebook: <br>
https://www.kaggle.com/joythabo33/99-accurate-cnn-that-detects-pneumonia/notebook

- Unit8 pneumonia git: <br>
https://github.com/unit8co/amld-workshop-pneumonia/tree/master/3_pneumonia


