from model import *
from data import *
from sklearn.model_selection import train_test_split
from utils import load_dataset

def main():
    # Load data
    global mask
    data, labels, masks = load_dataset(file_id)
    ############# CHANGE BELOW LINE WHEN USING MORE THAN ONE MASK #############
    mask = mask[0]
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(data, labels, masks, test_size=0.2, random_state=42)
    if save_test:
        np.save(os.path.join("data", f"{file_id}_Xtest.npy"), X_test)
        np.save(os.path.join("data", f"{file_id}_ytest.npy"), y_test)
    del data

    model = unet(input_size=(740,409,3))
    # model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(data, labels, masks, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, batch_size=2, nb_epoch=10, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

if __name__ == '__main__':
    main()