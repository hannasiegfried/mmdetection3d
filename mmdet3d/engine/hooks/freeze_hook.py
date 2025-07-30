from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

@HOOKS.register_module()
class FreezeHook(Hook):
    def __init__(self, freeze_epochs=5):
        self.freeze_epochs = freeze_epochs
        self._is_frozen = False

    def before_train_epoch(self, runner):
        current_epoch = runner.epoch
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        # Freeze layers
        if current_epoch < self.freeze_epochs:
            if not self._is_frozen:
                print(f"[FreezeModelHook] Freezing model at epoch {current_epoch}")
                for param in model.waveform_model.parameters():  # Adjust this to the module you want to freeze
                    param.requires_grad = False
                self._is_frozen = True
        else:
            if self._is_frozen:
                print(f"[FreezeModelHook] Unfreezing model at epoch {current_epoch}")
                for param in model.waveform_model.parameters():
                    param.requires_grad = True
                self._is_frozen = False
