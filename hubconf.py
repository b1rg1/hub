dependencies = ['torch']

def HUB_MODEL(*args, **kwargs):
    import torch
    model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)

    return model
