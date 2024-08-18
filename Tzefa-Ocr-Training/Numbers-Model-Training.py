import os
import os
import torch
import evaluate
import numpy as np
import pandas as pd
import glob as glob
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from zipfile import ZipFile
from tqdm.notebook import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from urllib.request import urlretrieve
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-stage1')
#model = VisionEncoderDecoderModel.from_pretrained(
    #'microsoft/trocr-base-handwritten'
#).to(device)

def read_image(image_path):
    """
    :param image_path: String, path to the input image.


    Returns:
        image: PIL Image.
    """
    image = Image.open(image_path).convert('RGB')
    return image


def ocr(image, processor, model):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.


    Returns:
        generated_text: the OCR'd text string.
    """
    # We can directly perform OCR on cropped images.
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cer_metric = evaluate.load('cer')

def compute_cer(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}
class CustomOCRDataset(Dataset):
    def __init__(self, datasetpath,root_dir, processor, max_target_length=128):
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.dataset=open(datasetpath).readlines()
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # The image file name.
        # The text (label).
        line = self.dataset[idx]
        line=line.split('.png')
        if(len(line)==1):
            line=line[0].split('.jpg')
            file_name = line[0] + '.jpg'
        else:
            file_name=line[0]+'.png'
        text=line[1]
        # Read the image, apply augmentations, and get the transformed pixels.
        image = Image.open(file_name).convert('RGB')
        image = train_transforms(image)
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        # Pass the text through the tokenizer and get the labels,
        # i.e. tokenized labels.
        labels = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_target_length
        ).input_ids
        # We are using -100 as the padding token.
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
if __name__ == '__main__':
    seed_everything(42)

    @dataclass(frozen=True)
    class TrainingConfig:
        BATCH_SIZE: int = 8
        EPOCHS: int = 5
        LEARNING_RATE: float = 0.00005


    @dataclass(frozen=True)
    class DatasetConfig:
        DATA_ROOT: str = r'C:\DataSets\Numbers'


    @dataclass(frozen=True)
    class ModelConfig:
        MODEL_NAME: str = "C:\Storage\models digits\checkpoint-3600"



    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=.3),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transfor.t
    ])
    processor = processor
    train_dataset = CustomOCRDataset(
        root_dir=os.path.join(DatasetConfig.DATA_ROOT, r'C:\DataSets\Numbers'),
        processor=processor,
        datasetpath=r"C:\DataSets\textfiles\numbertrain.txt"
    )
    valid_dataset = CustomOCRDataset(
        root_dir=os.path.join(r'C:\DataSets\Numbers2', r'C:\DataSets\Numbers2'),
        processor=processor,
        datasetpath=r"C:\DataSets\textfiles\numbervalid.txt"
    )
    model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
    model.to(device)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # Set Correct vocab size.
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    model.config.max_length = 20
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 20
    model.config.length_penalty = 2.0
    model.config.num_beams = 1
    optimizer = optim.AdamW(
        model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=0.0005
    )
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
        fp16=True,
        output_dir='C:\Storage\models digits',
        logging_strategy='steps',
        save_strategy='steps',
        save_steps=100,
        save_total_limit=5,
        report_to='tensorboard',
        num_train_epochs=TrainingConfig.EPOCHS
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_cer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator
    )
    res = trainer.train()

