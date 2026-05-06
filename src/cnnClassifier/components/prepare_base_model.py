import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

#In this class, we prepare the base model, which will accept the preparebasemodel config defined earlier. 
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

#download the vgg16 model from the Keras. How to download is given here #https://keras.io/api/applications/vgg/vgg_models/#vgg16-function
    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
#also saving this model
        self.save_model(path=self.config.base_model_path, model=self.model)


    #here you are doing the modification to the base downloaded model. 
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        #Freeze all :Use pretrained features only, Freeze partially ine-tune last layers, No freezing Full retraining
        if freeze_all:
            #Case 1: Freeze all layers: freeze each layer individually
            for layer in model.layers:
                model.trainable = False
                #Case 2: Partial Freezing
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

#Step 2: Add Custom Layers. Converts feature maps → 1D vector. Example (7, 7, 512) → (25088,)
#Final classification layer, units=classes is Number of output categories. Since it is 2, binary classification, so Softmax was used. 
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)
# Step 3: Creates full model by combining Original CNN base with new classification head. This is Transfer learning.  
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

#Step 4 model compile where SGD is used. Oftern ADAM is used instead of SGD for faster convergence, SGD is more suitable for fine tuning.
# #CategoricalCrossentropy() is used for multi class classification. Metrics us accuracy.  
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
#The _prepare_full_model function is responsible for constructing and compiling the transfer learning architecture, but it does not perform any training. 
# The update_base_model method acts as a pipeline step that invokes this function, stores the resulting model, and persists it to disk. 
# This separation aligns with modular ML pipeline design, where model preparation, training, and evaluation are handled as independent stages.”
#You will see later in this notebook in the pipeline that the _prepare_full_model is not called, it is actually the update_base_model which gets called and executes the _prepare_full_model
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)