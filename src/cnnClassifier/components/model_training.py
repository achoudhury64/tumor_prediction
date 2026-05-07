import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

#initialize training
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
#rescale=1./255 — Normalizes pixel values from [0, 255] → [0, 1]. Neural networks train faster and more stably with normalized inputs.
#validation_split=0.20 — 80/20 train/validation split done automatically by Keras.
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )
#This dict is passed to both flow_from_directory() calls (train & validation), so it defines how images are physically loaded and prepared.
#params_image_size comes from params.yaml and typically looks like: IMAGE_SIZE: [224, 224, 3]  # [Height, Width, Channels].

# The [:-1] slice drops the last element (channels), giving: just Height x Width. 
# it doesn't take channels because channels are determined separately by color_mode parameter (defaults to "rgb" = 3 channels). 
# Passing all 3 values would throw an error.

#Full Batch Gradient Descent     Mini-Batch (this code)      Stochastic (batch=1)
#───────────────────────────     ──────────────────────      ────────────────────
#Uses ALL data per update        Uses 16-32 images           Uses 1 image
#Stable but slow                 Balanced ✓                  Noisy, fast
#Can't fit in GPU RAM            Fits in GPU RAM ✓           Fits easily

#interpolation="bilinear"
#This controls the resizing algorithm when an image's original size ≠ target_size.
#What bilinear interpolation does:
#When scaling an image, new pixel values must be estimated. Bilinear looks at the 4 nearest neighbors and computes a weighted average based on distance:

#Why bilinear specifically for medical imaging?

#Preserves smooth intensity gradients in CT scans (important for tissue boundaries)
#Fast enough for real-time data loading
#Doesn't over-sharpen edges (bicubic can introduce ringing artifacts)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
#Why shuffle=False for validation? Because you want deterministic evaluation — shuffling validation data doesn't help 
# learning and makes metric comparison across epochs inconsistent. dataflow_kwargs was defined previously. 
#No augmentation for validation generator, only for training generator
#Validation generator — always plain, never augmented Built first, unconditionally. Takes the validation subset. 
# Shuffle is explicitly False — deterministic order is required for meaningful epoch-to-epoch metric comparison.
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
#if augmentation is true from Params.yaml, then Augmentation is done, else valid generated is used. 
# #What Augmentation Actually Does? It creates synthetic variations of your real images on-the-fly during training. 
# The model never sees the same version of an image twice, which forces it to learn features, not memorized pixel patterns.
#Original CT Scan
#      │
#      ├──► slightly rotated version    → batch 1
#      ├──► horizontally flipped        → batch 2  
#      ├──► zoomed in slightly          → batch 3
#      └──► shifted left + sheared      → batch 4

#Model sees 4 "different" images, but it's the same scan
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40, #Randomly rotates the image anywhere between -40° and +40°
                horizontal_flip=True, #Randomly mirrors the image left-to-right with 50% probability
                width_shift_range=0.2, #Randomly translates (slides) the image horizontally/vertically by up to 20% of total width/height
                height_shift_range=0.2,
                shear_range=0.2, #Applies a shear transformation — slants the image along an axis
                zoom_range=0.2, #Randomly zooms in or out by up to 20%
                **datagenerator_kwargs #this was unpacked from earlier
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
#the training model is simplified with diagram in the next cell
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

