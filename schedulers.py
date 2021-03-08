'''
File to store different keras schedulers for my models in models.py
'''

#This specific schedule code was modified from DnCNN GitHub: https://github.com/husqin/DnCNN-keras
def lr_schedule1(epoch):
    initial_lr = 0.001
    if epoch<=30:
        lr = initial_lr
    elif epoch<=60:
        lr = initial_lr/10
    elif epoch<=80:
        lr = initial_lr/20
    else:
        lr = initial_lr/20
    return lr

def lr_schedule2(epoch):
    initial_lr = 0.01
    if epoch<=10:
        lr = initial_lr
    elif epoch<=20:
        lr = initial_lr/10
    elif epoch<=40:
        lr = initial_lr/20
    else:
        lr = initial_lr/20
    #log('current learning rate is %2.8f' %lr)
    return lr