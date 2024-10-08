### CycleGAN.py
The class CycleGAN contains the implementation of the CycleGAN model that is the structure of generator and discriminator.

### CycleGANTraining.py
The class CycleGANTraining, subclass of keras.Model, overwrites the method train_step to perform the training of the model.

### image_processing.py
One function to load and pre-process images. Images can be downloaded from 'https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/', then organize them in directory such as train_A, train_B, test_A, test_B.

### main.py
Before running check the configuration values (number of images, size, epoch, learning rate...); this code performs the training of the model, save in json file the training loss and store in the directory model_checkpoint the parameters for each epoch.

### inference.ipynb
Load model from the directory model_checkpoint and perform image-translation on the images of the validation set; plot the training loss.
