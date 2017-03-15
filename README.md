# Traffic-sign-classifier
The goal of this project is to train a convolutional nueral net in tensorflow to classify traffic sign images using logistic regression. The original data set for this multi-class classification is available [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) although a pickled dataset provided by Udacity is used for this implemetation.
Note that even though certain types of image classification problems(OCR etc.) using DL has exceeded human accuracy,it is not the case yet with traffic signs under real world scenarios that include different lighting conditions, wear&tear on signage and other obstacles. 

Key steps of this implementation are:
* Explore and visualize the german traffic sign dataset
* Design a logistic classification architecture 
* Build a image processing pipeline to generalize the data
* Implement a DNN using tensorflow
* Train and validate the model
* Test the model on captured images 

---
### Code

The python notebook `traffic_sign_classifier.ipynb` implements the dataset visualization, processing pipeline and the tensorflow model model. Implementation consists of the following files located in the source directory

* traffic_sign_classifier.ipynb     -   Implements CNN model and processing pipeline   
* out_images                        -   Folder with additional test images 
* writeup.md                        -   You are reading it

### Data exploration and visualization

The traffic sign dataset that we use for the project consists of a more than 50,000 images classified into 43 classes. A description of the classes are included below for reference. Each traffic sign image is a resized RGB image of 32x32x3 pixels. The dataset is split into training, validation and test datasets as shown below.    

|  Traffic Sign Dataset                      |
|:------------------------------------------:|
|    Training Set      :  34799              |
|    Validation Set    :  4410               |     
|    Test Set          :  12630              |

Here is a random sample image from each of the 43 classes in the dataset. From these images, it is clear that the light conditions and clarity of the features in the image vary considerably. Looking at this a robust pre-processing pipeline is needed to accurately extract features from these images. As described later, the pre-processing consists of normalization and local contrast enhancement. There are a multitude of techniques in literature that can be further applied to enhance the quality of the dataset. This will be an interesting avenue to explore further.

![alt text](./writeup_images/image_random_sample.png)

Looking at the distribution of training set images by class, it is clear the distribution is not very uniform. There are certain classes of traffic signs that are underrespresented. Augmenting data in these underrespresented classes is nececessary to improve the overal test accuracy. Described below are images with 

![alt text](./writeup_images/image_class_hist.png)


### Model Architecture and Training

The model architecture is similar to the architecture proposed by Sermanet located [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). While this paper seems to be a little dated, it was a great starting point for the model architecture that resulted in > 95% test accuracy just with a few tweaks. Most of the parameters such as learning rates, epochs, convolution window sizes and sampling  etc. were tweaked empirically. An interesting area to explore is to research the current state of the art architectures and the performance being achieved.

As shown in the paper, the final architecture implemented here is "Lenet with feed-forward connections". The intial stages consists of 2 convolutional layers with 5x5 convolution windows. Relu activation and 2x2 max pooling is applied after each conv. layer. The next stages have three FC layers with dropout to predict the final classification. In addition, the ouputs of first conv. layer is fed-forward to the FC stage after processing through an addition maxpooling stage. The idea here is that the information available post the first stage(which is higher level features and shapes) is preserved and given more weighting in output classification. While this trick may have been helpful in 2011 due to limited capability of running larger networks, it is not entirely clear whether the same performance cannot be achieved today by just employing a larger network with more trainable parameters.

The processing pipeline consits of converting the RGB image to gray scale, normalization and local contrast adjustment. The dataset was split into training, validation and test sets with the images in each set shown above. Training and validation losses were monitored to ensure that the model is not overfitting the data. To better generalize, dropout was used in the FC layers toward the output. It was also observed that 150 epochs of training are needed before the validation losses to flatten out and not overfit. There is room for more optimization - especially in terms of augmenting the training dataset using flipping the images and selective shadowing etc.

The classifier achieves a validation accuracy of >97% and a test accuracy of >95% with the architecture used. It easily detects images the are reasonably clear , but often gives wrong predictions when the images were captured at complex angles, shadows or when signs are stacked. Performance in each of these scenarious can be further improved by augmenting the data and better processing techniques. While there are no plans to improve the model further, this was a good learning exercise to understand the importance of quality input data.

### Final Model 

The final model architecture is located in the file `traffic_sign_classifier.ipynb` and is shown below. As mentioned above, it consists of 2 convolution layers followed by 3 FC layers. Each convolution layers is followed by a Relu activation layers and a max pooling layer. The convolution windows are 5x5 and pooling windows are chosen to be 2x2. In addition, the output of the first stage (after pooling) is fed into first FC layer. 

```python
   from tensorflow.contrib.layers import flatten

   def LeNet(x, dr1, dr2):    
        # Arguments used for tf.truncated_normal, randomly defines variables 
        # for the weights and biases for each layer
        mu = 0
        sigma = 0.1
    
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 108), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(108))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # Activation.
        conv1 = tf.nn.relu(conv1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer 2: Convolutional. Output = 10x10x16.
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 108, 108), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(108))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
        # Activation.
        conv2 = tf.nn.relu(conv2)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten. Input = 5x5x16. Output = 400.
        fc0   = flatten(conv2)
    
        #Feedforward with or without additional 2nd stage subsampling
        fc0_multi = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        fc0_multi = flatten(fc0_multi)
        #fc0_multi = flatten(conv1)
        
        fc0 = tf.concat(1,[fc0,fc0_multi])
        #print(fc0.get_shape())
        # Without 2nd stage subsampling 14x14x108+5x5x108 = 21168+2700 = 23868
        # With additional 2nd stage subsampling 7x7x108+5x5x108 = 5292+2700 = 7992
    
    
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_W = tf.Variable(tf.truncated_normal(shape=(7992, 100), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(100))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
        # Activation.
        fc1    = tf.nn.relu(fc1)

        #Dropout
        fc1    = tf.nn.dropout(fc1,dr1)
    
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(100, 50), mean = mu, stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(50))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
        # Activation.
        fc2    = tf.nn.relu(fc2)

        #Dropout
        fc2    = tf.nn.dropout(fc2,dr2)
    
        # Layer 5: Fully Connected. Input = 84. Output = 43.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(50, 43), mean = mu, stddev = sigma))
        fc3_b  = tf.Variable(tf.zeros(43))
        logits = tf.matmul(fc2, fc3_W) + fc3_b
    
        return logits
```

Here is a visualization of network and output from the model that shows the parameters in each layer. As shown below, there are ~750K parameters that are trained in the network.

![alt text](./writeup_images/conv_net.png)

### Pre-processing Images

Again, like most machine learning problems, data processing - both augmenting data and pre-processing is critically important to get good performance out of the model. 

The first step involves converting the 32x32 RGB image into a grayscale image. 

```python
def pre_process(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
```

Next the images are normalized as shown below

```python
def normalize_img(image):
    norm_img = np.zeros(image.shape)
    return cv2.normalize(image,norm_img,alpha=0.0,beta=1.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
```
Experiments were also done with adding an additional processing step of improving local contrast in the images

```python
def exp_equalize(image):
    return exposure.equalize_adapthist(image)
```

Further areas to explore are in augmenting the data, especially the techniques below can be quite easily implemented. 
1. Flipping each image along the vertical axis
2. Changing the angle of the images

### Running the pipeline and testing 

![alt text](./writeup_images/image_pipeline.png)

The images are read in batches using a generator. Flipping the images is accomplished in the generator itself rather than processing it before hand. This eliminates loading all the images in memory at the beginning greatly reducing the memory requirements. 

Generator is implemented in the function `generator_images()`

After the data is augmented, total dataset comprised of ~18K images. The data set was split into training and validation sets at 80/20 ratio after random shuffling.

|  Dataset including left and right cameras |
|:-----------------------------------------:|
|    Total images   : 18537                 |
|    Training Set   : 14829                 |
|    Validation Set : 3708                  |                     

The network was run for 10 epochs anf Adam optimizer was used with a modified learning rate of 0.0001. This is observed  to result in slightly better validation performance than the default learning rate.

### Results

Here are the results obtained from images on the web. These test images were collected from google maps in Europe, cropped and resized to 32x32 pixels and fed to the model.


![alt text](./writeup_images/test_images.png)

![alt text](./writeup_images/test_images_preprocess.png)

![alt text](./writeup_images/test_image_results.png)
---

### Discussion and further work
This project is a good introduction to Keras and CNN's. Many improvements can be seen, mostly as a result of collecting better training data and augmenting. The CNN model seems quite robust and capable of learning complex non-linear functions as already demonstrated in the NVIDIA study mentioned above. Further work can be done in implementing a more complex network and run it on data collected in the real world.
