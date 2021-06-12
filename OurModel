from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop,Adam
import matplotlib.pyplot as plt


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
        batch_size=128,
        classes = ['airplanes','barrel','brain','cellphone','dolphin','electric_guitar','elephant','Faces','pyramid','sunflower'],
        class_mode='categorical')


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color

    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),

    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # 10 output neurons for 10 classes with the softmax activation
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['acc'])

total_sample=train_generator.n
n_epochs = 30

history = model.fit( train_generator, steps_per_epoch=int(total_sample/128), epochs=n_epochs,verbose=1)


#################### training accuracy #################

plt.figure(figsize=(7,4))
plt.plot([i+1 for i in range(n_epochs)],history.history['acc'],'-o',c='k',lw=2,markersize=9)
plt.grid(True)
plt.title("Training accuracy with epochs\n",fontsize=18)
plt.xlabel("Training epochs",fontsize=15)
plt.ylabel("Training accuracy",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


#################### training loss #################
plt.figure(figsize=(7,4))
plt.plot([i+1 for i in range(n_epochs)],history.history['loss'],'-o',c='k',lw=2,markersize=9)
plt.grid(True)
plt.title("Training loss with epochs\n",fontsize=18)
plt.xlabel("Training epochs",fontsize=15)
plt.ylabel("Training loss",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

################## Testing ###################

model.evaluate(test_generator)
