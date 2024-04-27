import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
import requests
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from pytorchtools import EarlyStopping
from transformers import BartTokenizer, T5Tokenizer,GPT2Tokenizer, GPT2LMHeadModel
from torchvision import transforms
from config import Config
import pickle
import clip
from torch.nn import functional as nnf
import math
from os.path import basename, dirname, join, isfile
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.nn.modules.activation import ReLU
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from tqdm import tqdm






#@title Caption prediction

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in tqdm(range(entry_count)):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0] 
    
    
    

#CUDA_VISIBLE_DEVICES=3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

class VXDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data,
    ):

        
        self.data=data        
        self.tokenizer = Config.TOKENIZER
        self.image_transform = transforms.Compose([
                                                  transforms.ToTensor(),                               
                                                  transforms.Resize((224,224)),
                                                  #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])
                                                
                                                      
        text_clip_feat=open("/home/krishanu_2021cs19/ACL2023/VLCybebully/Dataset/text_clip.pkl","rb")  #CLIP_textual_feature
        self.text_clip_feat=pickle.load(text_clip_feat)


        image_clip_feat=open("/home/krishanu_2021cs19/ACL2023/VLCybebully/Dataset/image_clip.pkl","rb")    #CLIP_visual_feature
        self.image_clip_feat=pickle.load(image_clip_feat)
  
    def __len__(self):
       
        return len(self.data)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_id=self.data["pid"][idx]
        
        
        vx_path = os.path.join("/home/krishanu_2021cs19/ACL2023/CMEx/VisualExplanations",img_id+".npy")
        v_path = os.path.join("/home/krishanu_2021cs19/ACL2023/VLCybebully/Bully_Images",img_id)
        
        sample = {
            "id":img_id,
            #"tokens":torch.zeros(20).to(device).long(),
            "encoder_input_ids":torch.zeros(128).to(device).long(),
            "encoder_attention_mask":torch.zeros(128).to(device).long(),
            "decoder_input_ids":torch.zeros(128).to(device).long(),
            "decoder_attention_mask":torch.zeros(128).to(device).long(),
            "text_clip_feat":torch.zeros(77).to(device).long(),
            "image_clip_feat":torch.zeros(3,224,224).to(device).float(),
            #"mask":torch.zeros(40).to(device).long(),
            "vx_label":torch.zeros(224,224).to(device).float(),
        }

        
        if os.path.exists(v_path) and os.path.exists(vx_path):
        
            vx_label = torch.tensor(np.load(vx_path)).to(device).float()
            
            
            try:
      
                
                text_clip_feat = self.text_clip_feat[img_id].to(device).long()
                image_clip_feat = self.image_clip_feat[img_id].to(device).long()
               
                #src = self.tokenizer.encode(str(self.data["labels"][idx]), return_tensors="pt")
                
                #tokens = torch.cat((src,torch.zeros(1,300-src.shape[1])),dim=1).reshape(-1).to(device).long()
                #mask = torch.cat((torch.ones(Config.PREFIX_LENGTH).to(device).long(),tokens.ge(0).long()),dim=0).reshape(-1).to(device).long()
     
                      
                src = self.tokenizer(str(self.data["text"][idx]), return_tensors="pt")
                
                src["input_ids"] = torch.cat((src["input_ids"],torch.zeros(1,300-src["input_ids"].shape[1])),dim=1).reshape(-1).to(device).long()
                src["attention_mask"] = torch.cat((src["attention_mask"],torch.zeros(1,300-src["attention_mask"].shape[1])),dim=1).reshape(-1).to(device).long()
                
         
                
                tgt = self.tokenizer(str(self.data["labels"][idx]), return_tensors="pt")
                
                tgt["input_ids"] = torch.cat((tgt["input_ids"],torch.zeros(1,300-tgt["input_ids"].shape[1])),dim=1).reshape(-1).to(device).long()
                tgt["attention_mask"] = torch.cat((tgt["attention_mask"],torch.zeros(1,300-tgt["attention_mask"].shape[1])),dim=1).reshape(-1).to(device).long()
                
                
                
                sample = {
                "id":img_id,
                #"tokens":tokens[:20].to(device).long(),
                "encoder_input_ids":src["input_ids"][:128].to(device).long(),
                "encoder_attention_mask":src["attention_mask"][:128].to(device).long(),
                "decoder_input_ids":tgt["input_ids"][:128].to(device).long(),
                "decoder_attention_mask":tgt["attention_mask"][:128].to(device).long(),
                "text_clip_feat":text_clip_feat.to(device).long(),
                "image_clip_feat":image_clip_feat.to(device).float(),
                #"mask":mask[:20+Config.PREFIX_LENGTH].to(device).long(),
                "vx_label":vx_label.to(device).float(),
                }
                
            except Exception as e:
                print(e)
            
            
            
            
        return sample
 
 
 



model = Config.MODEL.to(device)



train_data = pd.read_csv("/home/krishanu_2021cs19/ACL2023/VLCybebully/TextualExplanations/df_train.tsv",sep="\t")
val_data = pd.read_csv("/home/krishanu_2021cs19/ACL2023/VLCybebully/TextualExplanations/df_val.tsv",sep="\t")
test_data = pd.read_csv("/home/krishanu_2021cs19/ACL2023/VLCybebully/TextualExplanations/df_test.tsv",sep="\t")
 


#Loading VXDataset

vx_dataset_train = VXDataset(train_data)
dataloader_train = DataLoader(vx_dataset_train, batch_size=32,shuffle=False)

print("train_data loaded")

vx_dataset_val = VXDataset(val_data)
dataloader_val = DataLoader(vx_dataset_val, batch_size=32,shuffle=False)
print("validation_data loaded")


vx_dataset_test = VXDataset(test_data)
dataloader_test = DataLoader(vx_dataset_test, batch_size=1,shuffle=False)
print("test data loaded")


# Initialising hyperparameters


lr = 0.0001

clip_model,_ = clip.load("ViT-B/32", device=device)

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(list(model.parameters()),lr=lr)


exp_name = "EMNLP_MCHarm_GLAREAll_COVTrain_POLEval"

exp_path = "CMEx/"+exp_name


chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')

model.load_state_dict(torch.load(chk_file))



target = "/home/krishanu_2021cs19/ACL2023/CMEx/Predicted"
            


predictions = []
gold = []
use_beam_search = False

with torch.no_grad():
    for data in dataloader_test:
        sigmoid = nn.Sigmoid()
                    
                    
        with torch.no_grad():
            image_features = clip_model.encode_image(data["image_clip_feat"])
            text_features = clip_model.encode_text(data["text_clip_feat"])
        
        inputs={}
                  
        inputs['prefix'] = image_features.to(device).float() + text_features.to(device).float()
        #inputs['tokens'] = data['tokens'].to(device)
        #inputs['mask'] = data['mask'].to(device)
        inputs['input_ids'] = data['encoder_input_ids'].to(device)
        inputs['attention_mask'] = data['encoder_attention_mask'].to(device)
        #inputs['decoder_input_ids'] = data['decoder_input_ids'].to(device)
        inputs['labels'] = data['decoder_input_ids'].to(device)
        #inputs['image_features'] = data['image_features'].to(device)
        inputs['inp_image'] = data['image_clip_feat'].to(device)
        vx_label_test = data['vx_label'].to(device)
              
                    
        img_id = data["id"]
        
        prefix_embed = model.clip_project(inputs["prefix"]).reshape(1, Config.PREFIX_LENGTH, -1)
            
        seg_out = model.clip_seg(inp_image = inputs["inp_image"],conditional=inputs["prefix"])
        
                  
        out_idx = nn.Sigmoid()(seg_out[0].view(-1))
        
        out_idx = (out_idx>0.5).long()
        
        print(out_idx.shape)
        
        np.save(os.path.join(target,str(img_id[0])),out_idx.cpu().data.numpy())
        
        out_idx[out_idx==1]=225
        
        out_idx = out_idx.reshape(224,224)
        
        print(out_idx.shape)
        
        img = Image.fromarray(out_idx.cpu().data.numpy().astype(np.uint8))
        img.save(os.path.join(target,str(img_id[0])))
 
 
       
        generated_ids = model.t5_base.generate(input_ids=inputs["input_ids"],
                                       attention_mask=inputs["attention_mask"],
                                       # acoustic_input=acoustic_input,
                                       #clip_text_feat = clip_text_feat,
                                       #clip_img_feat = clip_img_feat,
                                       #att_map=att_map,
                                       image_features=prefix_embed,
                                      )
                          
    
                      
        generated_ids = generated_ids.detach().cpu().numpy()
        generated_ids = np.where(generated_ids != -100, generated_ids, Config.TOKENIZER.pad_token_id)
        decoded_preds = Config.TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
        
        labels = inputs["labels"].detach().cpu().numpy()
        labels = np.where(labels != -100, labels, Config.TOKENIZER.pad_token_id)
        decoded_labels = Config.TOKENIZER.batch_decode(labels, skip_special_tokens=True)
            
        predictions.extend(decoded_preds)
        gold.extend(decoded_labels)
        
        print(decoded_preds[0])
        print(decoded_labels[0])






test_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
test_df.to_csv(Config.SAVE_TEXT_PATH)

