import transformers
from transformers import BartTokenizer
from modeling_bart import BartForMultiModalGeneration
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration
from modeling_t5 import T5ForMultiModalGeneration
#from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class Config:
    MAX_LEN = 50
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 0.5e-5
    GPT_LEARNING_RATE = 0.5e-4
    #PROCESSOR = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    #MODEL = T5ForMultiModalGeneration.from_pretrained('t5-base', cross_attn_type=5,fusion_layer=True,use_img_trans=True)
    #MODEL = T5ForMultiModalGeneration.from_pretrained('t5-large', cross_attn_type=1,fusion_layer=False,use_img_trans=False)
    MODEL = BartForMultiModalGeneration.from_pretrained('facebook/bart-base',cross_attn_type=1,fusion_layer=True, use_img_trans=True)
    #MODEL = BartForMultiModalGeneration.from_pretrained('facebook/bart-large',cross_attn_type=5)
    #SAVE_PATH = "/home/krishanu_2021cs19/ACL2023/MultimodalBART/Results/T5-base/TEXTX_1.csv"
    SAVE_PATH = "/home/krishanu_2021cs19/ACL2023/MultimodalBART/Results/T5-large/TEXTX_1_only.csv"
    #SAVE_PATH = "/home/krishanu_2021cs19/ACL2023/MultimodalBART/Results/Bart-base/TEXTX.csv"
    #SAVE_PATH = "/home/krishanu_2021cs19/ACL2023/MultimodalBART/Results/Bart-large/TEXTX.csv"
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    CNN_MODEL_PATH = "Weights/cnn-text-classifier-v6-3-4-0.0005.pt"
    BERT_MODEL_PATH = "Weights/bert-style-classifier-20-64-5e-5.pt"
    GPT2_MODEL_PATH = "Weights-GPT2LM-20-8-5e-06.pt"
    TOKENIZER = BartTokenizer.from_pretrained('facebook/bart-base')
    #TOKENIZER = BartTokenizer.from_pretrained('facebook/bart-large')
    #TOKENIZER = T5Tokenizer.from_pretrained('t5-base')
    #TOKENIZER = T5Tokenizer.from_pretrained('t5-large')
    TRAINED_MODEL_PATH = 'CAST-BERT-Classifier-BT-Class-Rec-after-5-epoch-20-8-5e-06.pt'
    GPU_N = 0
