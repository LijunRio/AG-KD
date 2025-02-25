from torch.utils.data import DataLoader
from transformers import AutoProcessor
import torch
from dataset import VindrDataset,DetInstructDataset_vindr,VindrDataset_unkown,DetInstructDataset_vindr_Ukown
import pytorch_lightning as pl



class VinderDataLoaderManager(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        # Extract parameters from config
        self.img_root = config.get("img_root")
        self.annotation_csv = config.get("annotation_csv")
        self.batch_size = config.get("batch_size", 8)
        self.data_pct = config.get("data_pct", 1.0)
        self.num_workers = config.get("num_workers", 0)
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_definition = config["use_definition"]
        
        # Use the passed processor or initialize a default one


    def collate_fn(self,batch):
        # Unzip the batch into questions, answers, and images
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        images = [item['image'] for item in batch]
        tasks = [item['task'] for item in batch]
    
        return images,questions, answers, tasks
        
    
    def create_dataloader(self, split):
        """
        Creates a DataLoader for the given dataset split (train/val/test).
        
        Args:
            split (str): The dataset split ('train', 'val', or 'test').

        Returns:
            DataLoader: The DataLoader for the given split.
        """
        # Initialize the base dataset (MIMICDataset)
        base_dataset = VindrDataset(
            img_root=self.img_root,
            annotation_csv=self.annotation_csv,
            split=split,  # 'train', 'val', or 'test'
            data_pct=self.data_pct,
            transform=None
        )

        # Initialize the Multi_task_Instructer dataset
        # import pdb; pdb.set_trace()
        multi_task_dataset = DetInstructDataset_vindr(
            base_dataset=base_dataset,
            use_definition=self.use_definition
        )

        # Create DataLoader
        return DataLoader(
            multi_task_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,  # Use the custom collate function
            num_workers=self.num_workers,  # Adjust based on your system's capabilities
            shuffle=True if split == 'train' else False
        )
    
    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.
        """
        return self.create_dataloader(split='train')

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.
        """
        return self.create_dataloader(split='test')

    def test_dataloader(self):
        """
        Returns the DataLoader for the test set.
        """
        return self.create_dataloader(split='test')



class VinderDataLoaderManager_Ukown(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        # Extract parameters from config
        self.img_root = config.get("img_root")
        self.annotation_csv = config.get("annotation_csv")
        self.batch_size = config.get("batch_size", 8)
        self.data_pct = config.get("data_pct", 1.0)
        self.num_workers = config.get("num_workers", 0)
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print('❗use_definition',config["use_definition"])
        self.use_definition = config["use_definition"]
        
        # Use the passed processor or initialize a default one


    def collate_fn(self,batch):
        # Unzip the batch into questions, answers, and images
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        images = [item['image'] for item in batch]
        tasks = [item['task'] for item in batch]
    
        return images,questions, answers, tasks
        
    
    def create_dataloader(self, split):
        """
        Creates a DataLoader for the given dataset split (train/val/test).
        
        Args:
            split (str): The dataset split ('train', 'val', or 'test').

        Returns:
            DataLoader: The DataLoader for the given split.
        """
        # Initialize the base dataset (MIMICDataset)
        base_dataset = VindrDataset_unkown(
            img_root=self.img_root,
            annotation_csv=self.annotation_csv,
            split=split,  # 'train', 'val', or 'test'
            data_pct=self.data_pct,
            transform=None
        )

        # Initialize the Multi_task_Instructer dataset
        # import pdb; pdb.set_trace()
        multi_task_dataset = DetInstructDataset_vindr_Ukown(
            base_dataset=base_dataset,
            use_definition=self.use_definition
        )

        # Create DataLoader
        return DataLoader(
            multi_task_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,  # Use the custom collate function
            num_workers=self.num_workers,  # Adjust based on your system's capabilities
            shuffle=True if split == 'train' else False
        )
    
    def train_dataloader(self):
        """
        Returns the DataLoader for the training set.
        """
        return self.create_dataloader(split='train')

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation set.
        """
        return self.create_dataloader(split='test')

    def test_dataloader(self):
        """
        Returns the DataLoader for the test set.
        """
        return self.create_dataloader(split='test')
