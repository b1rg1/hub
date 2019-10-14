
def PYTORCH_MODEL(*args, **kwargs):
    dependencies = ['torch']
    import torch
    model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)

    return model

def TENSORFLOW_MODEL(*args, **kwargs):
    dependencies = ['tensorflow']
    import tensorflow as tf

    model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                              include_top=False,
                                              weights='imagenet')
    return model

def MY_MODEL(*args, **kwargs):
    
    return None
