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
RESİM_GENİŞLİK=128
RESİM_UZUNLUK=128
RESİM_BOYUT=(RESİM_GENİŞLİK,RESİM_UZUNLUK)
RESİM_KANALLAR=3

#eğitim verisinin hazırlanması
dosyaİsimleri = os.listdir("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train")
kategoriler = []
for dosyaİsmi in dosyaİsimleri:
    kategori = dosyaİsmi.split('.')[0]
    if kategori == 'dog':   #1 numaralı sınıf köpek
        kategoriler.append(1)
    else:
        kategoriler.append(0) #0 numaralı sınıf kedi
df = pd.DataFrame({
    'dosyaİsmi': dosyaİsimleri,
    'kategori': kategoriler
})
    
#verisayısına görsel olarak bakma
df['kategori'].value_counts().plot.bar()

#rastgele bir görüntünün seçilmesi
sample = random.choice(dosyaİsimleri)
image = load_img("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train/"+sample)
plt.imshow(image)

#modelin oluşturulması
from tensorflow.keras.models import Sequential #bu kısım değişti
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization #bu kısım değişti

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(RESİM_GENİŞLİK, RESİM_UZUNLUK, RESİM_KANALLAR)))
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
model.add(Dense(2, activation='softmax')) # 2 because we have cat and köpek classes
model.summary()

#modelin derlenmesi
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#verinin hazırlanması
df["kategori"] = df["kategori"].replace({0: 'kedi', 1: 'köpek'}) 
eğitim_df, doğrulama_df = train_test_split(df, test_size=0.20, random_state=42)
eğitim_df = eğitim_df.reset_index(drop=True)
doğrulama_df = doğrulama_df.reset_index(drop=True)

#katogirlere bakılması
eğitim_df['kategori'].value_counts().plot.bar()

#eğitim ve doğrulama verisinin hazırlanması
toplam_eğitim = eğitim_df.shape[0]
toplam_doğrulama = doğrulama_df.shape[0]
batch_size=15

#eğitim verilerinin çoğaltılması
eğitim_dataOluştur = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

eğitim_oluştur = eğitim_dataOluştur.flow_from_dataframe(
    eğitim_df, 
    "C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train/", 
    x_col='dosyaİsmi',
    y_col='kategori',
    target_size=RESİM_BOYUT,
    class_mode='categorical',
    batch_size=batch_size
)

#doğrulama verilerinin çoğaltılması
doğrulama_veriOluştur = ImageDataGenerator(rescale=1./255)
doğrulama_oluştur = doğrulama_veriOluştur.flow_from_dataframe(
    doğrulama_df, 
    "C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train/", 
    x_col='dosyaİsmi',
    y_col='kategori',
    target_size=RESİM_BOYUT,
    class_mode='categorical',
    batch_size=batch_size
)

#çoğaltılan veriye bakma
örnek_df = eğitim_df.sample(n=1).reset_index(drop=True)
örnek_oluştur = eğitim_dataOluştur.flow_from_dataframe(
    örnek_df, 
    "C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/train/", 
    x_col='dosyaİsmi',
    y_col='kategori',
    target_size=RESİM_BOYUT,
    class_mode='categorical'
)

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in örnek_oluştur:
        imge = X_batch[0]
        plt.imshow(imge)
        break
plt.tight_layout()
plt.show()

#modelin eğitilmesi
epochs=1
history = model.fit_generator(
    eğitim_oluştur, 
    epochs=epochs,
    validation_data=doğrulama_oluştur,
    validation_steps=toplam_doğrulama//batch_size,
    steps_per_epoch=toplam_eğitim//batch_size,
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
test_dosyaİsimleri = os.listdir("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/test1/")
test_df = pd.DataFrame({
    'dosyaİsmi': test_dosyaİsimleri
})
nb_samples = test_df.shape[0]

#test verilerinin çoğaltılması
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/test1/", 
    x_col='dosyaİsmi',
    y_col=None,
    class_mode=None,
    target_size=RESİM_BOYUT,
    batch_size=batch_size,
    shuffle=False
)

#tahmin işleminin yapılması
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

#₺ahmin işleminin hangi kategoriye ait olduğunu belirleme
test_df['kategori'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in eğitim_oluştur.class_indices.items())
test_df['kategori'] = test_df['kategori'].replace(label_map)

#tahminlerin değerlendirilmesi
sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    dosyaİsmi = row['dosyaİsmi']
    kategori = row['kategori']
    img = load_img("C:/Users/burak/Masaüstü/CNN Görüntü Sınıflandırma/kediKöpek/test1/"+dosyaİsmi, target_size=RESİM_BOYUT)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(dosyaİsmi + '(' + "{}".format(kategori) + ')' )
plt.tight_layout()
plt.show()





