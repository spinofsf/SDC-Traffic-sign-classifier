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

|  Traffic Sign Dataset                     |
|:-----------------------------------------:|
|    Training Set   : 34799                 |
|    Validation Set :  4410                 |     
|    Test Set       : 12630                 |

Here is a sample image from each class in the dataset

![alt text](./writeup_images/sample_image_class.png)

Looking at the distribution of images by class, we realize that it is quite uniform. There are certain classes of traffic signs that are underrespresented. It is clear that augmenting these data in these classes is neecessary to improve the overal test accuracy.

![alt text](./writeup_images/hist_class.png)


### Model Architecture and Training

The model architecture is similar to the architecture proposed by Sermanet located [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). While this paper seems to be dated, it was a great starting point for the model architecture that resulted in > 97% test accuracy with a few tweaks. Most of the parameters such as learning rates, epochs, convolution window sizes and sampling  etc. were  Further work can be done to adapt the model to the current state of the art in pattern classification.  

However, since our track and lane conditions are much simpler, the depth of the network and the nodes at each layer were reduced. As described below, the final model consists of 4 convolutional layers with 3x3 convolution windows. Relu activation and 2x2 max pooling is applied after each conv. layer. Finally 3 FC layers with dropout are utilized to estimate the output of steering angle. Most of the parameters such as window sizes, learning rate were finalized based on empirical data. 

The augmented data set was split into training and validation sets. Training and validation losses were monitored to ensure that the model is not overfitting the data. To better generalize, the driving data that was collected was augmented to reduce driving biases associated with the data set. Also, dropout was used in the dense layers toward the output. It was also observed that 10 epochs of training are sufficient to run the car reasonably well in autonomous mode. There is room for more optimization - both in terms of augmenting the data and the model if needed.

The car runs easily at the default speed setting of 9 mph set in `drive.py`. It also runs well at an increased speed setting of 15mph without crossing either of the lane boundaries. While there is a bit of moving sideways between the lanes and during the edges, this was primarily due to how the data was captured. The original data was captured at the fastest speed and did not necessarily keep the car always centered. This is another optimization that can be investigated.

### Final Model 

The final model architecture is located in the file `behavioural cloning.ipynb` and is shown below. It consists of 4 convolution layers followed by 3 FC layers. Each convolution layers is followed by a Relu activation layers and a max pooling layer. The convolution windows are 3x3 and pooling windows are chosen to be 2x2. A lamda layer takes the cropped input images and normalizes them before passing them through the conv. layers.

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



```python
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 70, 320, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 68, 318, 16)   448         lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 68, 318, 16)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 34, 159, 16)   0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 157, 24)   3480        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 32, 157, 24)   0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 16, 78, 24)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 14, 76, 32)    6944        maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 14, 76, 32)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 7, 38, 32)     0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 36, 64)     18496       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 5, 36, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 2, 18, 64)     0           activation_4[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2304)          0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 300)           691500      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 300)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           30100       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            1010        dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 751,989
Trainable params: 751,989
Non-trainable params: 0
____________________________________________________________________________________________________

```

### Pre-processing Images
```python
def pre_process(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def normalize_img(image):
    norm_img = np.zeros(image.shape)
    return cv2.normalize(image,norm_img,alpha=0.0,beta=1.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def exp_equalize(image):
    return exposure.equalize_adapthist(image)
```












Data was captured from the simulator in training mode and augmented. Total data set includes
1. three laps of center driving on the original track  
2. two laps of driving in reverse and 
3. one lap of recovery. 

Data collection is oen of the most important parts of this project. One of the experiments that was done was to capture the data while driving the car at the maximum speed which meant that the corners were not taken at the middle of the road, but closer to the edges like in the real world. This results in the car behaving very similarly in autonomous mode as well. The car comes close to the edges while taking a turn but stays within lanes. 

While there is a lot of straight line driving in the training mode that results in neutral steering angle, this data was still kept in the dataset without reducing it. Since this seems to be a valid real world scenario (lot of straight driving as opposed to curves), effort was made to keep the training data as-is and generalize the model using other techniques instead (reverse driving, lr flip, dropout).

Data was augmented in two ways
1. Including images from both the left and right cameras in the data set. Steering angle correction was left and right cameras was kept at +/-0.2 degrees. 
2. Flipping each image along the vertical axis

```python
    camera_adjust_angle = 0.2

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)    
        for row in reader:
            steering_center = float(row[3])
            steering_left = steering_center + camera_adjust_angle
            steering_right = steering_center - camera_adjust_angle        
```

Flipping images is implemented using the numpy function `fliplr()`

```python
    def flip_imgs(orig_imgs, orig_meas):
        new_imgs = [np.fliplr(image) for image in orig_imgs]
        new_meas = [-1*meas for meas in orig_meas]
        return new_imgs,new_meas
```

All the images are then cropped on the top and bottom portions of the image so that only the road section is passed through the model

```python
    def crop_imgs(orig_imgs, crop_htop=70, crop_hbot=140): 
        new_imgs = [image[crop_htop:crop_hbot,:,:] for image in orig_imgs]        
        return new_imgs
```
Every image captured from the center of the camera is thus processed to generate 6 total images. Shown below are the original images from the left, center and right cameras captured in simulation mode. Note that the steering angles are empirically set to an offset of 0.2deg in either direction. The next row shows the result of flipping each image across the horizontal axis. THe bottom two rows show the cropped versions so that only the road area that is of interest is preserved.

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
