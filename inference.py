import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from CycleGAN import CycleGAN
from CycleGANTraining import CycleGANTraining
from image_processing import load_data

def load_and_plot_losses(json_file):
    # Load the JSON file
    with open(json_file, 'r') as file:
        losses_history = json.load(file)
    
    # Convert dictionary entries to NumPy arrays
    loss_G = np.array(losses_history['G_loss'])
    loss_F = np.array(losses_history['F_loss'])
    loss_DX = np.array(losses_history['D_X_loss'])
    loss_DY = np.array(losses_history['D_Y_loss'])

    ls_loss_G = np.array(losses_history['ls_loss_G'])
    cyc_loss_G = np.array(losses_history['cycle_loss_G'])
    id_loss_G = np.array(losses_history['id_loss_G'])

    ls_loss_F = np.array(losses_history['ls_loss_F'])
    cyc_loss_F = np.array(losses_history['cycle_loss_F'])
    id_loss_F = np.array(losses_history['id_loss_F'])

    epochs = np.arange(1, len(loss_G) + 1)

    # Plot the losses
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, loss_G, label='loss-G', color='tab:blue')
    plt.plot(epochs, loss_F, label='loss-F', color='tab:green')
    plt.plot(epochs, loss_DX, label='loss-DX', color='tab:red')
    plt.plot(epochs, loss_DY, label='loss-DY', color='tab:orange')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('CycleGAN - Training loss', size=20)
    plt.legend(fontsize=10)
    plt.xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig('CycleGAN.png')
    plt.show()

    # Generator G
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, loss_G, label='loss-G', color='tab:blue')
    plt.plot(epochs, ls_loss_G, label='ls-loss-G', color='tab:blue', linestyle='-.', linewidth=1)
    plt.plot(epochs, cyc_loss_G, label='cyc-loss-G', color='tab:blue', linestyle='--', linewidth=1)
    plt.plot(epochs, id_loss_G, label='id-loss-G', color='tab:blue', linestyle=':', linewidth=1)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Generator G - Training loss', size=20)
    plt.legend(fontsize=10)
    plt.xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig('CycleGAN_G.png')
    plt.show()
    
    # Generator F
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, loss_F, label='loss-F', color='tab:green')
    plt.plot(epochs, ls_loss_F, label='ls-loss-F', color='tab:green', linestyle='-.', linewidth=1)
    plt.plot(epochs, cyc_loss_F, label='cyc-loss-F', color='tab:green', linestyle='--', linewidth=1)
    plt.plot(epochs, id_loss_F, label='id-loss-F', color='tab:green', linestyle=':', linewidth=1)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Generator F - Training Loss', size=20)
    plt.legend(fontsize=10)
    plt.xticks(np.arange(0, 101, 10))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig('CycleGAN_F.png')
    plt.show()

# ========================================================================

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# load data for validation
n_samples = 120
data_dir_A = 'data/horse2zebra/test_horses'
data_dir_B = 'data/horse2zebra/test_zebras'
data_A, data_B, dataset = load_data(data_dir_A, data_dir_B, IMG_SIZE, BATCH_SIZE, n_samples, info=False)

# plot training loss
load_and_plot_losses('history.json')

# instantiate CycleGAN object
print("[INFO] initializing the CycleGAN model...")
my_model = CycleGAN(IMG_HEIGHT, IMG_WIDTH)
# initialize generator and discriminator
disc_X = my_model.discriminator()
disc_Y = my_model.discriminator()
gen_G = my_model.generator()
gen_F = my_model.generator()

# build the CycleGAN training model and compile it
print("[INFO] building and compiling the CycleGAN training model...")
# Build the CycleGAN model
cycleGAN  = CycleGANTraining(
    generator_G = gen_G, 
    generator_F = gen_F, 
    discriminator_X = disc_X, 
    discriminator_Y = disc_Y
)

# load weights
print("[INFO] loading weights the CycleGAN training model...")
weight_file = "model_checkpoints/cyclegan_checkpoints.100"
cycleGAN.load_weights(weight_file)


images_X = []
images_Y = []
for image_X in data_A.take(n_samples):
    images_X.append(image_X)
for image_Y in data_B.take(n_samples):
    images_Y.append(image_Y)

# horse -> zebra
i = 1
for img in images_X:
    
    fake_y = cycleGAN.gen_G(img, training=False)[0].numpy()
    cycle_x = cycleGAN.gen_F(np.expand_dims(fake_y, axis=0), training=False)[0].numpy()

    # Rescaling from [-1,+1] to [0,255]
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    fake_y = (fake_y * 127.5 + 127.5).astype(np.uint8)
    cycle_x = (cycle_x * 127.5 + 127.5).astype(np.uint8)

    print(f'{i} IMAGE')
    
    plt.figure(figsize=(15, 5))
    
    # First subplot: img
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Input Image', fontsize=20, fontweight='bold')
    plt.axis('off')
    
    # Second subplot: fake_y
    plt.subplot(1, 3, 2)
    plt.imshow(fake_y)
    plt.title('Fake Image', fontsize=20, fontweight='bold')
    plt.axis('off')
    
    # Third subplot: cycle_x
    plt.subplot(1, 3, 3)
    plt.imshow(cycle_x)
    plt.title('Cycled Image', fontsize=20, fontweight='bold')
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'images\horse2zebra\horse2zebra_{i}.png', bbox_inches='tight')
    plt.show()

    i += 1

# zebra -> horse
i = 1
for img in images_Y:
    
    fake_x = cycleGAN.gen_F(img, training=False)[0].numpy()
    cycle_y = cycleGAN.gen_F(np.expand_dims(fake_x, axis=0), training=False)[0].numpy()

    # Rescaling from [-1,+1] to [0,255]
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
    fake_x = (fake_x * 127.5 + 127.5).astype(np.uint8)
    cycle_y = (cycle_y * 127.5 + 127.5).astype(np.uint8)

    print(f'{i} IMAGE')
    
    
    plt.figure(figsize=(15, 5))
    
    # First subplot: img
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Input Image', fontsize=20, fontweight='bold')
    plt.axis('off')
    
    # Second subplot: fake_x
    plt.subplot(1, 3, 2)
    plt.imshow(fake_x)
    plt.title('Fake Image', fontsize=20, fontweight='bold')
    plt.axis('off')
    
    # Third subplot: cycle_y
    plt.subplot(1, 3, 3)
    plt.imshow(cycle_y)
    plt.title('Cycled Image', fontsize=20, fontweight='bold')
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'images\zebra2horse\zebra2horse_{i}.png', transparent=True, bbox_inches='tight')
    plt.show()

    i += 1

