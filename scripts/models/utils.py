from tensorflow.keras import backend as K

def masked_MSE(mask):
    '''
    MSE, only over masked areas
    '''
    def loss_fn(y_true, y_pred):
        for yt in y_true:  # for each example in the batch
            yt = yt[mask == True]
        for yp in y_pred:
            yp = yp[mask == True]
        loss_val = K.mean(K.square(K.abs(y_pred - y_true)))
        return loss_val
    return loss_fn

def masked_MSE_multiple_masks(y_true, y_pred):
    ''' 
    MSE, only over masked areas. ALLOWS FOR INDIVIDUAL MASKS, embedded in:
        real mask: labels[i][:,:,2]
        sim - real mask: labels[i][:,:,3]
        We will take the loss inside the fake masks and outside of the real masks (i.e. ch 4)
    '''
    for i in range(len(y_true)):  # for each example in the batch
        yt = y_true[i]
        yp = y_pred[i]
        sim_minus_real_mask = yt[:, :, 3]  # mask diff in ch 4
        print("yt", yt.shape)
        print("yp", yp.shape)
        print("mask", sim_minus_real_mask.shape)

        yt = yt[sim_minus_real_mask == True]
        yp = yp[sim_minus_real_mask == True]
    # take loss over masked areas
    loss_val = K.mean(K.square(K.abs(y_pred - y_true[:, :, :, :2])))
    return loss_val