
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

#CUDA_VISIBLE_DEVICES=3

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
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
dataloader_test = DataLoader(vx_dataset_test, batch_size=32,shuffle=False)
print("test data loaded")


# Initialising hyperparameters


lr = 0.0001

clip_model,_ = clip.load("ViT-B/32", device=device)

#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(list(model.parameters()),lr=lr)


exp_name = "EACL2024"

exp_path = "CMEx/"+exp_name

#model= nn.DataParallel(model, device_ids=[0,1,2,3,4])

#Training Starts
#chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')

#model.load_state_dict(torch.load(chk_file))




 
def train_model(model, patience, n_epochs):
    epochs = n_epochs
#     clip = 5

    
    train_loss_list=[]
    val_loss_list=[]
    
        # initialize the experiment path
    Path(exp_path).mkdir(parents=True, exist_ok=True)
    # initialize early_stopping object
    
    chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=chk_file)


    model.train()
    for i in range(epochs):


        total_loss_train = 0
        total_train = 0
        

        for data in dataloader_train:
            sigmoid = nn.Sigmoid()
                        
                        
            with torch.no_grad():
                image_features = clip_model.encode_image(data["image_clip_feat"])
                text_features = clip_model.encode_text(data["text_clip_feat"])
                

                    
            inputs = {}

            #inputs['prefix'] = image_features.to(device).float() + text_features.to(device).float()
            inputs['prefix'] = image_features.to(device).float() 
            #inputs['tokens'] = data['tokens'].to(device)
            #inputs['mask'] = data['mask'].to(device)
            inputs['inp_image'] = data['image_clip_feat'].to(device)
            inputs['input_ids'] = data['encoder_input_ids'].to(device)
            inputs['attention_mask'] = data['encoder_attention_mask'].to(device)
            #inputs['decoder_input_ids'] = data['decoder_input_ids'].to(device)
            inputs['labels'] = data['decoder_input_ids'].to(device)
            #inputs['image_features'] = data['image_features'].to(device)
            vx_label_train = data['vx_label'].to(device)
            
        

            model.zero_grad()
           
            
            loss_sum,seg_out = model(**inputs)
            
            #logits = sum_out.logits[:, Config.PREFIX_LENGTH - 1: -1]
            #loss_sum = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), inputs["tokens"].flatten(), ignore_index=0)
  
            
            output = seg_out[0].view(-1)
            vx_label_train = vx_label_train.view(-1)
            
            

            loss_seg = criterion(sigmoid(output), vx_label_train)
            
            #loss = loss_sum
            loss = loss_seg
            #loss = loss_seg + loss_sum

           
            loss.backward()

            optimizer.step()

            total_loss_train += loss.item()
            total_train += vx_label_train.size(0)

        
       
        train_loss = total_loss_train/total_train
        model.eval()
        
        print(train_loss)
#         total_acc_val = 0
        total_loss_val = 0
        total_val = 0

        with torch.no_grad():
            for data in dataloader_val:
                sigmoid = nn.Sigmoid()
                    
                                       
                with torch.no_grad():
                    image_features = clip_model.encode_image(data["image_clip_feat"])
                    text_features = clip_model.encode_text(data["text_clip_feat"])
                    

                        
                inputs = {}
                    
                #inputs['prefix'] = image_features.to(device).float() + text_features.to(device).float()
                inputs['prefix'] = image_features.to(device).float()
                #inputs['tokens'] = data['tokens'].to(device)
                #inputs['mask'] = data['mask'].to(device)
                inputs['input_ids'] = data['encoder_input_ids'].to(device)
                inputs['attention_mask'] = data['encoder_attention_mask'].to(device)
                #inputs['decoder_input_ids'] = data['decoder_input_ids'].to(device)
                inputs['labels'] = data['decoder_input_ids'].to(device)
                #inputs['image_features'] = data['image_features'].to(device)
                inputs['inp_image'] = data['image_clip_feat'].to(device)
                vx_label_val = data['vx_label'].to(device)
              
                    
                    
                model.zero_grad()
                
                       
                #sum_out,seg_out = model(**inputs)
                val_loss_sum,seg_out = model(**inputs)
                
                #logits = sum_out.logits[:, Config.PREFIX_LENGTH - 1: -1]
                #val_loss_sum = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), inputs["tokens"].flatten(), ignore_index=0)
      
                
                output = seg_out[0].view(-1)
                vx_label_val = vx_label_val.view(-1)
                
            

                val_loss_seg = criterion(sigmoid(output), vx_label_val)
                
                #val_loss = val_loss_sum
                val_loss = val_loss_seg
                #val_loss = val_loss_seg + val_loss_sum
                
                    
              
                total_val += vx_label_val.size(0)
                                
                total_loss_val += val_loss.item()


        val_loss = total_loss_val/total_val

     
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        #early_stopping.best_score=None
        early_stopping(val_loss, model)
        
        #print("Saving model...") 
        #torch.save(model.state_dict(), os.path.join(exp_path, "final.pt"))
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            

        
        print(f'Epoch {i+1}: train_loss: {train_loss:.10f} | val_loss: {val_loss:.10f}')
        model.train()
        torch.cuda.empty_cache()
        
    # load the last checkpoint with the best model
    #model.load_state_dict(torch.load(chk_file))
    #model.load_state_dict(torch.load(os.path.join(exp_path,"final.pt")))
    
    return  model, train_loss_list, val_loss_list, i
        




n_epochs = 50

patience = 50

#chk_file1 = os.path.join(exp_path, 'checkpoint_text_'+exp_name+'.pt')
#chk_file2 = os.path.join(exp_path, 'checkpoint_image_'+exp_name+'.pt')

#model_sum.load_state_dict(torch.load(chk_file1))
#model_seg.load_state_dict(torch.load(chk_file2))

#model.load_state_dict(torch.load(chk_file))
model, train_loss_list, val_loss_list, epoc_num = train_model(model, patience, n_epochs)

 
