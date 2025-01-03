from lightning.pytorch.callbacks import Callback

class CustomCallback(Callback):
    def __init__(self):
        pass
        
    def on_train_start(self, trainer, pl_module):
        pass
        
    def on_train_end(self, trainer, pl_module):
        pass 