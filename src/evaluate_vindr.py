import os
import torch
import pytorch_lightning as pl
from transformers import AdamW, get_scheduler, AutoModelForCausalLM, AutoProcessor
from Florence2.preprocessor_florence2 import Florence2Processor  
from peft import LoraConfig, get_peft_model
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from datetime import datetime
from dataloader import VinderDataLoaderManager
import yaml
from tools import render_inference_results2
import wandb
from checkpoint_callback import CustomModelCheckpoint
from peft import PeftModel, PeftConfig
from transformers import AutoConfig
import supervision as sv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CLASSES = ['Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Cardiomegaly', 'Nodule or Mass', 'Lung Opacity', 'Other lesion', 'Pleural effusion', 'ILD', 'Infiltration', 'Calcification', 'Consolidation', 'Atelectasis', 'Rib fracture', 'Mediastinal shift', 'Enlarged PA', 'Pneumothorax', 'Emphysema', 'Lung cavity', 'Lung cyst', 'Clavicle fracture', 'Edema']
CLASSES = [cls.lower() for cls in CLASSES]

class FlorenceLightningModel(pl.LightningModule):
    def __init__(self, model, processor, lr=1e-6, num_training_steps=None):
        super(FlorenceLightningModel, self).__init__()
        self.model = model
        self.processor = processor
        self.lr = float(lr)
        self.num_training_steps = num_training_steps
        self.test_outputs = []
        self.valid_outputs = []


    def training_step(self, batch, batch_idx):
        images,questions,answers, tasks = batch
        inputs = self.processor(
            text=questions, 
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        labels = self.processor.tokenizer(
            text=answers,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False
        ).input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size = len(images), sync_dist=True)  # Log to progress bar
        torch.cuda.empty_cache()  
        return loss


    def validation_step(self, batch, batch_idx):
        images,questions,answers, tasks = batch
        # import pdb; pdb.set_trace()
        inputs = self.processor( 
            text=questions, 
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        from tools import evluate_resultsevluate_results
        batch_results = evluate_resultsevluate_results(model=self.model, inputs=inputs,processor=self.
                                                       processor, answers=answers, images=images,batch_idx=batch_idx,
                                                       questions=questions)
        batch_data = [
            wandb.Image(img, caption=cap) 
            for img,cap in zip(batch_results['res_samples'], batch_results['captions'])
        ]
        self.logger.experiment.log({
            "generated_images": batch_data,
            "epoch": self.current_epoch,  # 当前 epoch
            "batch_idx": batch_idx,       # 当前 batch 索引
            "step": self.current_epoch  # Ensure the step is consistent
        })
        self.valid_outputs.append(batch_results)

        return batch_results

    def on_validation_epoch_end(self):
        all_predictions = []
        all_targets = []

        for batch_result in self.valid_outputs:
            all_predictions.extend(batch_result["predictions"])
            all_targets.extend(batch_result["targets"])


        mean_average_precision = sv.MeanAveragePrecision.from_detections(
            predictions=all_predictions, 
            targets=all_targets
        )

        self.log("val/mAP_50_95", mean_average_precision.map50_95)
        self.log("val/mAP_50", mean_average_precision.map50)
        self.log("val/mAP_75", mean_average_precision.map75)

    
    def test_step(self, batch, batch_idx):
        images,questions,answers, tasks = batch
        # import pdb; pdb.set_trace()
        inputs = self.processor( 
            text=questions, 
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        from tools import evluate_resultsevluate_results
        batch_results  = evluate_resultsevluate_results(model=self.model, inputs=inputs,processor=self.processor, answers=answers, images=images,batch_idx=batch_idx, questions=questions)
        batch_data = [
            wandb.Image(img, caption=cap) 
            for img,cap in zip(batch_results['res_samples'], batch_results['captions'])
        ]
        self.logger.experiment.log({
            "generated_images": batch_data,
            "epoch": self.current_epoch, 
            "batch_idx": batch_idx,       
            "step": self.current_epoch  
        })
        self.test_outputs.append(batch_results)

        return batch_results


        
    def on_test_epoch_end(self):
        all_predictions = []
        all_targets = []

        # import pdb; pdb.set_trace()
        for batch_result in self.test_outputs:
            all_predictions.extend(batch_result["predictions"])
            all_targets.extend(batch_result["targets"])

        res_csv = []
        for i in range (len(all_predictions)):
            pred = all_predictions[i]
            tgt = all_targets[i]
            # import pdb; pdb.set_trace()
            tgt_bbox = tgt.xyxy.tolist()
            pred_bbox = pred.xyxy.tolist()
            tgt_class_id = tgt.class_id.tolist()
            pred_class_id = pred.class_id.tolist()

            tgt_class_name = tgt['class_name'].tolist()
            pred_class_name = pred['class_name'].tolist() if pred['class_name'] is not None else [[]]
            row = {'tgt_bbox':tgt_bbox, 'pred_bbox':pred_bbox, 'tgt_class_id':tgt_class_id, 'pred_class_id':pred_class_id, 'tgt_class_name':tgt_class_name, 'pred_class_name':pred_class_name}
            res_csv.append(row)
        import pandas as pd
        res_csv = pd.DataFrame(res_csv)
        res_csv.to_csv('../res/our_vindr_res.csv')


        mean_average_precision = sv.MeanAveragePrecision.from_detections(
            predictions=all_predictions, 
            targets=all_targets
        )


        print("mAP_50_95:", mean_average_precision.map50_95)
        print("mAP_50:", mean_average_precision.map50)
        print("mAP_75:", mean_average_precision.map75)
        self.log("test/mAP_50_95", mean_average_precision.map50_95)
        self.log("test/mAP_50", mean_average_precision.map50)
        self.log("test/mAP_75", mean_average_precision.map75)
        print("Mean Average Precision:\n", mean_average_precision)

    


    def configure_optimizers(self):
        print('self.lr:', self.lr)
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        # num_training_steps = self.trainer.max_epochs * len(self.dataset.dataset)

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]
    

    


def main():
    args = parse_args() 
    # import pdb; pdb.set_trace()
    print('\n🧪 \033[1;36mEXPERIMENT STARTED\033[0m') 
    print('📌 \033[1;33mWhether Using Definition as Prompt:\033[0m', 
        f'\033[1;32m{args.definition}\033[0m' if args.definition else '\033[1;31mFalse\033[0m')  
    print('📌 \033[1;33mPercentage of Data Used:\033[0m', f'\033[1;32m{args.data_pct}\033[0m')
    print('📌 \033[1;33mOutput Directory:\033[0m', f'\033[1;32m{args.output_dir}\033[0m')
    config_path = "../configs/experiment.yaml"
    config = load_config(config_path)

    

    if args.definition:
        name = f"test_with_definition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        name = f"test_without_definition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        

    wandb_logger = WandbLogger(
        project=config['trainer']['project_name'],
        entity=config['trainer']['entity_name'],
        name=name
    )
    wandb_logger.log_hyperparams({"use_definition": args.definition, "data_pct": args.data_pct}) # add
    # CHECKPOINT = "microsoft/Florence-2-base-ft"
    CHECKPOINT = args.checkpoint
    print('📌 \033[1;33mCheckpoint Path:\033[0m', f'\033[1;32m{args.checkpoint}\033[0m')
    REVISION = 'refs/pr/6'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
    # processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    
    ## Initialize the model
    config_model = AutoConfig.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
    config_model.vision_config.model_type = "davit"
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, config=config_model,revision=REVISION).to(DEVICE)
    processor = Florence2Processor.from_pretrained("./Florence2")
    processor.image_processor.size = config['model']['processor']['image_size']
    processor.image_processor.crop_size = config['model']['processor']['crop_size']

   
    config['dataset']['vindr']['data_pct'] = args.data_pct
    data_loader_manager = VinderDataLoaderManager({
        "img_root": config['dataset']['vindr']['img_root'],
        "annotation_csv": config['dataset']['vindr']['annotation_csv'],
        "batch_size": config['trainer']['train_batch_size'],
        "data_pct": config['dataset']['vindr']['data_pct'],  # Use the entire dataset for testing
        "num_workers": config['trainer']['num_workers'],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "processor": None,  # Use default processor
        "use_definition": args.definition,
    })


    train_dataloader = data_loader_manager.train_dataloader()
    val_dataloader = data_loader_manager.test_dataloader()

    dataset_size = len(train_dataloader.dataset)
    num_training_steps = (dataset_size + config['trainer']['train_batch_size'] - 1) // config['trainer']['train_batch_size']

    lightning_model = FlorenceLightningModel(model=model, processor=processor, lr=config['trainer']['learning_rate'], num_training_steps=num_training_steps)

    if config['trainer']['checkpoint_dir'] is not None:
        os.makedirs(config['trainer']['checkpoint_dir'], exist_ok=True)

    config['trainer']['checkpoint_dir'] = args.output_dir
    custom_checkpoint_callback = CustomModelCheckpoint(
        dirpath=config['trainer']['checkpoint_dir'],
        filename='model-{epoch}-{step}',
        save_top_k=2,  # Save top 2 models based on the monitored metric
        monitor='val/mAP_50',  # Monitor a different metric (e.g., val_accuracy)
        mode='max',  # Mode for monitoring (min for loss, max for accuracy)
        # every_n_train_steps=500,  # every_n_train_steps >= save_top_k*val_check_interval
        save_embedding_layers=True,  # Save the embedding layers
        verbose=True  # Set to True to log when checkpoints are saved
    )


    trainer = Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator="gpu",
        devices=1,
        # devices=[0,1],
        # strategy="ddp" if config['trainer']['ddp'] else None,
        log_every_n_steps=200,
        logger=wandb_logger,
        callbacks=[custom_checkpoint_callback]
    )


    trainer.test(lightning_model, val_dataloader)


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for unknown detection tasks.")
    parser.add_argument('--definition', action='store_true', help='Enable definition usage (default: False).')
    parser.add_argument('--data_pct', type=float, default=1.0, help='Percentage of data to use (default: 1.0).')
    parser.add_argument('--output_dir', type=str, default="../outputs/model_checkpoint_vindr", help='Path to the checkpoint to use (default: None).')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to use (default: None).')
    return parser.parse_args()

if __name__ == "__main__":
    main()
