
#from data_builder import SummaryDataModule
#from t5 import T5MultiModal
#from multi_modal_model import BartMultiModal
import torch
import torch.nn as nn
import os
import numpy as np
#from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from pytorchtools import EarlyStopping
from transformers import BartTokenizer, T5Tokenizer
from torchvision import transforms
from modeling_bart import BartForMultiModalGeneration
from modeling_t5 import T5ForMultiModalGeneration
from config import Config
#from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'


SOURCE_MAX_LEN = 1024
TARGET_MAX_LEN = 50
MAX_UTTERANCES = 25

ACOUSTIC_DIM = 154
ACOUSTIC_MAX_LEN = 600

VISUAL_DIM = 2048
VISUAL_MAX_LEN = 49

BATCH_SIZE = 1
MAX_EPOCHS = 20

BASE_LEARNING_RATE = 3e-5 
NEW_LEARNING_RATE = 3e-5 
# BASE_LEARNING_RATE=7e-5
# NEW_LEARNING_RATE=7e-5
WEIGHT_DECAY = 1e-4

NUM_BEAMS = 4
EARLY_STOPPING = True
NO_REPEAT_NGRAM_SIZE = 3

EARLY_STOPPING_THRESHOLD = 5

gen_kwargs = {
    'num_beams': NUM_BEAMS,
    'max_length': TARGET_MAX_LEN,
    'early_stopping': EARLY_STOPPING,
    'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE
}

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
                                                  transforms.Resize((352,352)),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])
        
        
         
 
        
  
    def __len__(self):
       
        return len(self.data)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_id=self.data["pid"][idx]
        
        
        vx_path = os.path.join("/home/krishanu_2021cs19/ACL2023/VLCybebully/VisualExplanations",img_id+".npy")
        v_path = os.path.join("/home/krishanu_2021cs19/ACL2023/VLCybebully/Bully_Images",img_id)
        
        sample = {
            "id":img_id,
            #"input_ids":torch.zeros(77).to(device).long(),
            #"attention_mask":torch.zeros(77).to(device).long(),
            "encoder_input_ids":torch.zeros(128).to(device).long(),
            "encoder_attention_mask":torch.zeros(128).to(device).long(),
            "decoder_input_ids":torch.zeros(77).to(device).long(),
            "decoder_attention_mask":torch.zeros(77).to(device).long(),
            #"pixel_values":torch.zeros(3,352,352).to(device).float(),
            "image_features":torch.zeros(3,352,352).to(device).float(),
            "vx_label":torch.zeros(352,352).to(device).float(),
        }

        
        if os.path.exists(v_path) and os.path.exists(vx_path):
        
            vx_label = torch.tensor(np.load(vx_path)).to(device).float()
            
            
            try:
                image = Image.open(v_path).convert('RGB')
                
                img_transform = torch.tensor(self.image_transform(np.array(image))).to(device).float()
                
                #inputs = self.processor(text=[str(self.data["text"][idx])],images=[image], padding=True, return_tensors="pt")
                
                #print(str(self.data["text"][idx]))
                
                #inputs["input_ids"] = torch.cat((inputs["input_ids"],torch.zeros(1,300-inputs["input_ids"].shape[1])),dim=1).reshape(-1).to(device).long()
                #inputs["attention_mask"] = torch.cat((inputs["attention_mask"],torch.zeros(1,300-inputs["attention_mask"].shape[1])),dim=1).reshape(-1).to(device).long()
                
                src = self.tokenizer(str(self.data["text"][idx]), return_tensors="pt")
                
                src["input_ids"] = torch.cat((src["input_ids"],torch.zeros(1,300-src["input_ids"].shape[1])),dim=1).reshape(-1).to(device).long()
                src["attention_mask"] = torch.cat((src["attention_mask"],torch.zeros(1,300-src["attention_mask"].shape[1])),dim=1).reshape(-1).to(device).long()
                
         
                
                tgt = self.tokenizer(str(self.data["labels"][idx]), return_tensors="pt")
                
                tgt["input_ids"] = torch.cat((tgt["input_ids"],torch.zeros(1,300-tgt["input_ids"].shape[1])),dim=1).reshape(-1).to(device).long()
                tgt["attention_mask"] = torch.cat((tgt["attention_mask"],torch.zeros(1,300-tgt["attention_mask"].shape[1])),dim=1).reshape(-1).to(device).long()
                
                
                sample = {
                "id":img_id,
                #"input_ids":inputs["input_ids"][:77].to(device).long(),
                #"attention_mask":inputs["attention_mask"][:77].to(device).long(),
                "encoder_input_ids":src["input_ids"][:128].to(device).long(),
                "encoder_attention_mask":src["attention_mask"][:128].to(device).long(),
                "decoder_input_ids":tgt["input_ids"][:77].to(device).long(),
                "decoder_attention_mask":tgt["attention_mask"][:77].to(device).long(),
                #"pixel_values":inputs["pixel_values"].reshape(3,352,352).to(device).float(),
                "vx_label":vx_label.to(device).float(),
                "image_features":img_transform.to(device).float()
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


lr = 3e-5



#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=lr)


exp_name = "EMNLP_MCHarm_GLAREAll_COVTrain_POLEval_6"

exp_path = "GENX/Bart-base/"+exp_name


chk_file = os.path.join(exp_path, 'checkpoint_'+exp_name+'.pt')

model.load_state_dict(torch.load(chk_file))


predictions = []
gold = []
with torch.no_grad():
    for data in dataloader_test:
    
        inputs={}

        #inputs['input_ids'] = data['encoder_input_ids'].to(device)
        input_ids = data['encoder_input_ids'].to(device)
        #inputs['attention_mask'] = data['encoder_attention_mask'].to(device)
        attention_mask = data['encoder_attention_mask'].to(device)
        #inputs['decoder_input_ids'] = data['decoder_input_ids'].to(device)
        labels = data['decoder_input_ids'].to(device)
        #inputs['image_features'] = data['image_features'].to(device)
        image_features = data['image_features'].to(device)
        vx_label_train = data['vx_label'].to(device)
        
        #for key in gen_kwargs.keys():
        #    inputs[key]= gen_kwargs[key].to(device)
        
        print(gen_kwargs)
        
        
        generated_ids = model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       # acoustic_input=acoustic_input,
                                       #clip_text_feat = clip_text_feat,
                                       #clip_img_feat = clip_img_feat,
                                       #att_map=att_map,
                                       image_features=image_features,
                                       **gen_kwargs)
                          
    
                      
        generated_ids = generated_ids.detach().cpu().numpy()
        generated_ids = np.where(generated_ids != -100, generated_ids, Config.TOKENIZER.pad_token_id)
        decoded_preds = Config.TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)
        
        labels = labels.detach().cpu().numpy()
        labels = np.where(labels != -100, labels, Config.TOKENIZER.pad_token_id)
        decoded_labels = Config.TOKENIZER.batch_decode(labels, skip_special_tokens=True)
            
        predictions.extend(decoded_preds)
        gold.extend(decoded_labels)
        
        print(decoded_preds[0])
        #print(decoded_labels[0])


torch.cuda.empty_cache() 



test_df = pd.DataFrame(list(zip(gold, predictions)), columns=['actual_explanation', 'predicted_explanation'])
test_df.to_csv(Config.SAVE_PATH)


