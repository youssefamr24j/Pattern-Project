from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(batch_size=128,
                                    directory=r"/content/drive/MyDrive/Dataset_objects/Dataset_without_aug/train",
                                    target_size=(200, 200), 
                                    subset="training",        
                                    classes = ['airplanes','barrel','brain','cellphone','dolphin','electric_guitar','elephant','Faces','pyramid','sunflower'],
                                    class_mode='categorical')

test_generator = train_datagen.flow_from_directory(r"/content/drive/MyDrive/Dataset_objects/Dataset_without_aug/test",
        target_size=(200, 200),
        batch_size=50,
        classes = ['airplanes','barrel','brain','cellphone','dolphin','electric_guitar','elephant','Faces','pyramid','sunflower'],
        class_mode='categorical')
model = tf.keras.models.Sequential([
    # Here, we use a larger 11 x 11 window to capture objects. At the same
    # time, we use a stride of 4 to greatly reduce the height and width of
    # the output. Here, the number of output channels is much larger than
    # that in LeNet
    tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                            activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    # Make the convolution window smaller, set padding to 2 for consistent
    # height and width across the input and output, and increase the
    # number of output channels
    tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                            activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    # Use three successive convolutional layers and a smaller convolution
    # window. Except for the final convolutional layer, the number of
    # output channels is further increased. Pooling layers are not used to
    # reduce the height and width of input after the first two
    # convolutional layers
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                            activation='relu'),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                            activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                            activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Flatten(),
    # Here, the number of outputs of the fully-connected layer is several
    # times larger than that in LeNet. Use the dropout layer to mitigate
    # overfitting
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # Output layer. Since we are using Fashion-MNIST, the number of
    # classes is 10, instead of 1000 as in the paper
    tf.keras.layers.Dense(10)
])


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

total_sample=train_generator.n
n_epochs = 30

model.fit( train_generator, steps_per_epoch=int(total_sample/128), epochs=n_epochs,verbose=1)
model.evaluate(test_generator)
model.summary()
