## CycleGAN.py: 
the class CycleGAN contains the implementation of the CycleGAN model that is the structure of generator and discriminator.

## CycleGANTraining.py: 
the class CycleGANTraining, subclass of keras.Model, overwrite the method train_step to perform the training of the model.

## image_processing.py: 
one function to load and pre-process images.

## main.py: 
before running check the configuration values (number of images, size, epoch, learning rate...); this code perform the training of the model, save in json file the training loss and store in the directory model_checkpoint the parameters for each epoch.

## Inference.ipynb
load model from the directory model_checkpoint and perform image transaltion on the images of the validation set; plot the training loss.