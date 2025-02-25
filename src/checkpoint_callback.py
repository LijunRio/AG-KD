
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil
import os
class CustomModelCheckpoint(ModelCheckpoint): 
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    def __init__(self, *args, save_embedding_layers=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_embedding_layers = save_embedding_layers
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # import pdb; pdb.set_trace()
        # checkpoint = super().on_save_checkpoint(trainer, pl_module, checkpoint)
        base_output_dir = self.dirpath  
        epoch = trainer.current_epoch  # To match your previous code where epoch starts at 1
        step = trainer.global_step
        output_dir = os.path.join(base_output_dir, f"model-epoch={epoch}-step={step}")
        os.makedirs(output_dir, exist_ok=True)

        # Save model and processor
        pl_module.model.save_pretrained(output_dir)
        pl_module.processor.save_pretrained(output_dir)
        # # print(f"Custom saving: Model and processor saved for epoch {epoch}-{step} at {output_dir}")
        # subfolders = [f.path for f in os.scandir(base_output_dir) if f.is_dir()]
        # checkpoint_files = [os.path.splitext(f.path)[0] for f in os.scandir(base_output_dir) if f.is_file() and f.name.endswith('.ckpt')]
        # checkpoint_files.append(str(output_dir))
        # if len(subfolders) > self.save_top_k:
        #     # import pdb; pdb.set_trace()
        #     for subfolder in subfolders:
        #         if subfolder not in checkpoint_files:
        #             # print(f"Removing subfolder: {subfolder}")
        #             shutil.rmtree(subfolder)  
        return None  