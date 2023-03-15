import tensorflow as tf
import keras

import os
import random
import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
import pathlib
import argparse

import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
from model_settings import GAN, Generator, Discriminator
from utils import normalize_images
#######################
SHAPE = [476,476,3]

BATCH_SIZE = 8
EPOCHS = 15
noise_level = 1
val_size = 125

root_folder = r"" # Folder to store output results
seed_file = r"" # Seed .npy file from 'gan_training'
clustering_folder = r"" # Folder where unsupervised clustering results are stored
global_val_images = r"" # Random images from global GEE production file
checkpoint_path = r"" # Desired checkpoint file from 'gan_training'; re-loads model better than model.load()

RANDOM_SEED = 101
random.seed(RANDOM_SEED)
AUTOTUNE = tf.data.AUTOTUNE

latent_dim = 512

#####################

if not os.path.isdir(root_folder):
    os.mkdir(root_folder)

for class_num in range(18):
    CURRENT_EPOCH = 0
    try:
        print(f"STARTING CLASS {class_num} -----------------------------------------")
        seed = np.load(seed_file)
        input_folder = os.path.join(clustering_folder,f"class_{class_num}"
        output_folder = os.path.join(root_folder,f"ft_class_{class_num}")

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        data_dir = pathlib.Path(input_folder)
        import_ds = tf.keras.utils.image_dataset_from_directory(
          data_dir,
          labels=None,
          shuffle=True,
          image_size=(476,476),
          seed=RANDOM_SEED,
          batch_size=None)

        data_augmentation = keras.Sequential(
                        [keras.layers.RandomFlip("horizontal_and_vertical"),
                         keras.layers.GaussianNoise(noise_level)])

        train_ds = import_ds.shuffle(512)\
                    .map(lambda x: normalize_images(data_augmentation(x, training=True)),num_parallel_calls=AUTOTUNE)\
                    .batch(BATCH_SIZE).prefetch(AUTOTUNE)
        val_ds = import_ds.take(val_size).map(lambda x: normalize_images(x)).batch(1)
        
        generator = Generator(SHAPE)
        discriminator = Discriminator(SHAPE)

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005,beta_1=0.5)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        
        class SaveImageCallback(keras.callbacks.Callback):
            def __init__(self):
                super(SaveImageCallback, self).__init__()

            def generate_and_save_images(self, generator, epoch, test_image):
                epoch = epoch+1
                predictions = generator(test_image, training=False)
                fig = plt.figure(figsize=(12, 12))
                for i in range(predictions.shape[0]):
                    plt.subplot(5, 5, i+1)
                    arr = predictions[i, :, :, :]*255
                    arr = arr.numpy().astype("uint8")
                    plt.imshow(arr)
                    plt.axis('off')
                plt.subplots_adjust(hspace=0.01, wspace=0.01)
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder,'image_at_epoch_{:04d}.png'.format(epoch)))
                plt.close()

            def on_epoch_end(self, epoch, logs=None):
                global CURRENT_EPOCH
                global train_ds
                CURRENT_EPOCH = epoch+1
                self.generate_and_save_images(generator,epoch,seed)
                noise_calc = noise_level-(noise_level*CURRENT_EPOCH/(EPOCHS))*2
                if noise_calc <= 0:
                    noise_calc = 0
                data_augmentation = keras.Sequential(
                            [keras.layers.RandomFlip("horizontal_and_vertical"),
                             keras.layers.GaussianNoise(noise_calc)])
                train_ds = import_ds.shuffle(512)\
                                .map(lambda x: normalize_images(data_augmentation(x, training=True)),num_parallel_calls=AUTOTUNE)\
                                .batch(BATCH_SIZE).prefetch(AUTOTUNE)
                if CURRENT_EPOCH == EPOCHS:
                    checkpoint2.save(os.path.join(output_folder,f"class_{class_num}_ft"))
        
        def generate_stats(time):
            real_results = [discriminator(image,training=False).numpy()[0][0] for image in val_ds.as_numpy_iterator()]
            fake_results = [discriminator(generator(tf.random.normal([1, latent_dim]),training=False),training=False).numpy()[0][0] for i in range(val_size)]
            global_results = []
            for path in os.listdir(global_val_images):
                image = Image.open(os.path.join(global_val_images,path))
                image = keras.layers.CenterCrop(SHAPE[0], SHAPE[1])(np.asarray(image))
                global_results.append(discriminator(np.expand_dims(normalize_images(image),0),training=False).numpy()[0][0])
            df_r = pd.DataFrame(real_results,columns=['Discriminator Prediction'])
            df_r['Type'] = f'Class {class_num} Images (Real)'
            df_f = pd.DataFrame(fake_results,columns=['Discriminator Prediction'])
            df_f['Type'] = 'Generated Localized Images (Fake)'
            df_g = pd.DataFrame(global_results,columns=['Discriminator Prediction'])
            df_g['Type'] = 'Global Images (Real)'
            df = pd.concat([df_r,df_f,df_g])
            df['Time'] = time
            return df

        checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
        checkpoint2 = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
        checkpoint.restore(checkpoint_path).expect_partial()
        # for layer in discriminator.layers[:7]:
        #     layer.trainable = False
        
        model = GAN(generator, discriminator, latent_dim)
        model.compile(generator_optimizer, discriminator_optimizer, loss_fn)

        df_before = generate_stats('Before Fine Tuning')
        model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[SaveImageCallback()])
        df_after = generate_stats('After Fine Tuning')

        df = pd.concat([df_before,df_after])
        hist = sns.displot(df,x='Discriminator Prediction',hue='Type',col='Time',multiple='stack')
        plt.xlim(0,1);
        plt.xticks(np.arange(0,1,0.1));
        hist.fig.subplots_adjust(top=.85)
        hist.fig.suptitle(f"Fine Tuning Kernel Density Estimation Comparison for Class {class_num}");
        plt.savefig(os.path.join(root_folder,f"fine_tune_class_{class_num}.png"))
        
    except Exception as e:
        print(f"Class {class_num} passed due to {e}")
        pass
    

# ################################
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Generate xy locs from file names.')
#     parser.add_argument("--class_num")
#     args = parser.parse_args()
#     main(args.class_num)