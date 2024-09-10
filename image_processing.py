import matplotlib.pyplot as plt
import tensorflow as tf

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0

def load_data(data_dir_A, data_dir_B, img_size, batch_size, max_images, info=True):
    
    AUTOTUNE = tf.data.AUTOTUNE
    buffer_size = 256
    
    print()
    print(f'Loading data from {data_dir_A}')
    data_A = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir_A,
        label_mode=None,
        image_size=img_size,
        batch_size=batch_size
    )

    print()
    print(f'Loading data from {data_dir_B}')
    data_B = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir_B,
        label_mode=None,
        image_size=img_size,
        batch_size=batch_size
    )
    print()

    if info == True:
        images_A = []
        images_B = []
        for image_A in data_A.take(25):
            images_A.append(image_A[0].numpy().astype("uint8"))
        for image_B in data_B.take(25):
            images_B.append(image_B[0].numpy().astype("uint8"))

            
        plt.figure(figsize=(10, 10))
        for i, image in enumerate(images_A):  
            plt.subplot(5, 5, i + 1)
            plt.imshow(image)
            plt.axis('off')
        plt.suptitle(data_dir_A, fontsize=20, fontweight='bold')
        plt.show()

        plt.figure(figsize=(10, 10))
        for i, image in enumerate(images_B):  
            plt.subplot(5, 5, i + 1)
            plt.imshow(image)
            plt.axis('off')
        plt.suptitle(data_dir_B, fontsize=20, fontweight='bold')
        plt.show()

    # Limit the number of images
    if max_images != None:
        data_A = data_A.take(max_images // batch_size)
        data_B = data_B.take(max_images // batch_size)
        
    # Image rescaling from [0, 255] to [-1,1]
    data_A = (data_A.map(normalize_img).cache().prefetch(buffer_size=AUTOTUNE))
    data_B = (data_B.map(normalize_img).cache().prefetch(buffer_size=AUTOTUNE))

    dataset = tf.data.Dataset.zip((data_A, data_B)).cache().prefetch(buffer_size=AUTOTUNE)
    
    print(f'Dataset with ({len(data_A)},{len(data_B)}) images successfully loaded')
    print()

    return data_A, data_B, dataset
