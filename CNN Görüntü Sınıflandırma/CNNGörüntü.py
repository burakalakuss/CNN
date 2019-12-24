#gerekli kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
#klasörün gösterilmesi
print(os.listdir("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek"))

#gerekli sabitlerin belirlenmesi
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

#eğitim verisinin hazırlanması
filenames = os.listdir("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':   #1 numaralı sınıf köpek
        categories.append(1)
    else:
        categories.append(0) #0 numaralı sınıf kedi
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
    
#verisayısına görsel olarak bakma
df['category'].value_counts().plot.bar()

#rastgele bir görüntünün seçilmesi
sample = random.choice(filenames)
image = load_img("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train/"+sample)
plt.imshow(image)

#modelin oluşturulması
from tensorflow.keras.models import Sequential #bu kısım değişti
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization #bu kısım değişti

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes
model.summary()

#modelin derlenmesi
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#verinin hazırlanması
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

#katogirlere bakılması
train_df['category'].value_counts().plot.bar()

#eğitim ve doğrulama verisinin hazırlanması
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

#eğitim verilerinin çoğaltılması
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

#doğrulama verilerinin çoğaltılması
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

#çoğaltılan veriye bakma
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

#modelin eğitilmesi
epochs=1
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
)

#oluşturulan modelin kaydedilmesi
model.save_weights("model1.h5")

#eğitim ve doğrulama verisinin görüntülenmes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

#test verisinin hazırlanması
test_filenames = os.listdir("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/test1/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

#test verilerinin çoğaltılması
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/test1/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

#tahmin işleminin yapılması
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

#₺ahmin işleminin hangi kategoriye ait olduğunu belirleme
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)

#tahminlerin değerlendirilmesi
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/test1/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()





