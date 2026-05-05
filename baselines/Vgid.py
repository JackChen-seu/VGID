import copy
import os
import sys
from collections import defaultdict, Counter
import numpy as np

import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel
sys.path.append(('../'))
sys.path.append(('../../'))
from datasets import load_dataset, Dataset
import random
import torch
import os
import json
from torch.utils.data import Subset
import argparse
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, \
    LlavaNextForConditionalGeneration, LlavaNextProcessor, Idefics2ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
from data_process.data_preprocess_ic import Vanilla_LLaVA_Dataset, train_collate_fn_llava, train_collate_fn, \
    train_collate_fn_idefics, LLAVA_multimodal_Dataset
import matplotlib.pyplot as plt
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer, TrainingArguments
# from trl import SFTConfig, SFTTrainer
import random
from torch.utils.data import Subset
from torch.nn import functional as F
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
# Example usage:
def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if args.model_id.startswith("llava"):
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
            # quantization_config=bnb_config,
            # cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/llava-1.5-7b-hf",
        )
        processor = AutoProcessor.from_pretrained(args.model_id)

    elif args.model_id.startswith("HuggingFaceM4"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # Load LLAVA Next model and processor
        print("Loading idefics2 model...")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/idfics2-8b",
        )
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            do_image_splitting=False
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")


    # Additional processor configuration if necessary
    processor.tokenizer.padding_side = "right"  # Ensure right padding
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    return model, processor



######################### Accelerate Version ################################

def add_random_noise_to_image(image):
    """
    Replace pixel values with completely random noise.
    
    Args:
        pixel_values: Tensor of shape (batch_size, channels, height, width)
    
    Returns:
        noisy_pixel_values: Tensor with random noise
    """
    if image is None:
        return None
    
    # Generate random noise in the same range as the input
    random_noise = torch.rand_like(image)
    
    return random_noise

def main(args):

    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)
    model_vanilla = copy.deepcopy(model)  # Teacher model (vanilla model)
    model_vanilla.eval()  # Set teacher to evaluation mode
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Resize token embeddings to match the tokenizer
    # model.resize_token_embeddings(len(processor.tokenizer))
    # model_vanilla.resize_token_embeddings(len(processor.tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
        model_vanilla.resize_token_embeddings(len(processor.tokenizer))

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # model.add_adapter(lora_config)
    # model.enable_adapters()
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup

    # dataset = Vanilla_LLaVA_Dataset_baseline(json_dir=profile_dir, image_dir=image_base_path, flatten=False)
    # print(f"Dataset size (profiles): {len(dataset)}")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    forget_df = pd.read_parquet(forget_parquet_file)
    retain_df = pd.read_parquet(retain_parquet_file)

    multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
    multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)


    if args.model_id.startswith("llava"):
        # Prepare the training dataloaders
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args)
        )
        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args)
        )
    elif args.model_id.startswith("HuggingFaceM4"):
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
        )
        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    accelerator = Accelerator()
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = 4  # Adjust based on memory
        model.gradient_checkpointing_enable()
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader_forget) + len(train_dataloader_retain)) * args.num_epochs,
    )

    # Prepare model, optimizer, and scheduler with accelerator
    model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler
    )

    # Create iterators for both dataloaders
    forget_iter = iter(train_dataloader_forget)
    retain_iter = iter(train_dataloader_retain)
    
    # Calculate max steps per epoch (use the larger one)
    max_steps = min(len(train_dataloader_forget), len(train_dataloader_retain))

    # Training loop with interleaved forget and retain batches
    for epoch in range(args.num_epochs):
        model.train()
        total_loss_forget = 0
        total_loss_retain = 0
        
        # Reset iterators at the start of each epoch
        forget_iter = iter(train_dataloader_forget)
        retain_iter = iter(train_dataloader_retain)
        
        progress_bar = tqdm(total=max_steps * 2, desc=f"Epoch {epoch + 1}")
        
        # Interleaved training: process one forget batch, then one retain batch
        for step in range(max_steps):
            # Process forget batch
            try:
                batch_forget = next(forget_iter)
            except StopIteration:
                # Reset iterator if exhausted
                forget_iter = iter(train_dataloader_forget)
                batch_forget = next(forget_iter)
            
            input_ids, attention_mask, pixel_values, labels = batch_forget
            noisy_pixel_values = add_random_noise_to_image(pixel_values)
            
            if args.gradient_accumulation:
                with accelerator.accumulate(model):
                    outputs_student = model(input_ids=input_ids, attention_mask=attention_mask,
                                           pixel_values=pixel_values, labels=labels)
                    with torch.no_grad():
                        outputs_teacher = model_vanilla(input_ids=input_ids, attention_mask=attention_mask,
                                                        pixel_values=noisy_pixel_values, labels=labels)
                    kl_div_forget = F.kl_div(
                        F.log_softmax(outputs_student.logits, dim=-1),
                        F.softmax(outputs_teacher.logits, dim=-1),
                        reduction='batchmean'
                    )
                    loss_forget = kl_div_forget
                    weighted_loss_forget = args.alpha * (loss_forget / accumulation_steps)
                    accelerator.backward(weighted_loss_forget)
                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                outputs_student = model(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, labels=labels)
                with torch.no_grad():
                    outputs_teacher = model_vanilla(input_ids=input_ids, attention_mask=attention_mask,
                                                    pixel_values=noisy_pixel_values, labels=labels)
                kl_div_forget = F.kl_div(
                    F.log_softmax(outputs_student.logits, dim=-1),
                    F.softmax(outputs_teacher.logits, dim=-1),
                    reduction='batchmean'
                )
                loss_forget = kl_div_forget
                weighted_loss_forget = args.alpha * loss_forget
                accelerator.backward(weighted_loss_forget)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss_forget += loss_forget.item()
            progress_bar.update(1)
            progress_bar.set_postfix({
                'forget_loss': total_loss_forget / (step + 1),
                'retain_loss': total_loss_retain / (step + 1) if step > 0 else 0
            })
            
            # Process retain batch
            try:
                batch_retain = next(retain_iter)
            except StopIteration:
                # Reset iterator if exhausted
                retain_iter = iter(train_dataloader_retain)
                batch_retain = next(retain_iter)
            
            input_ids, attention_mask, pixel_values, labels = batch_retain
            
            if args.gradient_accumulation:
                with accelerator.accumulate(model):
                    outputs_current = model(input_ids=input_ids, attention_mask=attention_mask,
                                            pixel_values=pixel_values, labels=labels)
                    with torch.no_grad():
                        outputs_original = model_vanilla(input_ids=input_ids, attention_mask=attention_mask,
                                                         pixel_values=pixel_values, labels=labels)
                    kl_div = F.kl_div(
                        F.log_softmax(outputs_current.logits, dim=-1),
                        F.softmax(outputs_original.logits, dim=-1),
                        reduction='batchmean'
                    )
                    loss_retain = outputs_current.loss + kl_div
                    weighted_loss_retain = args.beta * (loss_retain / accumulation_steps)
                    accelerator.backward(weighted_loss_retain)
                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                outputs_current = model(input_ids=input_ids, attention_mask=attention_mask,
                                        pixel_values=pixel_values, labels=labels)
                with torch.no_grad():
                    outputs_original = model_vanilla(input_ids=input_ids, attention_mask=attention_mask,
                                                     pixel_values=pixel_values, labels=labels)
                kl_div = F.kl_div(
                    F.log_softmax(outputs_current.logits, dim=-1),
                    F.softmax(outputs_original.logits, dim=-1),
                    reduction='batchmean'
                )
                loss_retain = kl_div
                weighted_loss_retain = args.beta * loss_retain
                accelerator.backward(weighted_loss_retain)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss_retain += loss_retain.item()
            progress_bar.update(1)
            progress_bar.set_postfix({
                'forget_loss': total_loss_forget / (step + 1),
                'retain_loss': total_loss_retain / (step + 1)
            })
        
        # Step the learning rate scheduler after each epoch
        lr_scheduler.step()
        
        print(f"Epoch {epoch + 1} - Forget Loss: {total_loss_forget / max_steps}")
        print(f"Epoch {epoch + 1} - Retain Loss: {total_loss_retain / max_steps}")
        epoch_save_dir = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_save_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(epoch_save_dir)


    # Save the final model after training
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")

if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for loss_forget")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for loss_retain")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")

    args = parser.parse_args()

    # Call main function
    main(args)
