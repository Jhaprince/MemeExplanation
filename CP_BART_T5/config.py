import transformers
from transformers import BartTokenizer, T5Tokenizer,GPT2Tokenizer, GPT2LMHeadModel
from model import ClipCaptionModel
import clip

class Config:
    MAX_LEN = 50
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 0.5e-5
    GPT_LEARNING_RATE = 0.5e-4
    #TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
    TOKENIZER = T5Tokenizer.from_pretrained('t5-base')
    #TOKENIZER = BartTokenizer.from_pretrained('facebook/bart-base')
    PREFIX_LENGTH = 40
    MODEL = ClipCaptionModel(prefix_length = PREFIX_LENGTH)
    SAVE_TEXT_PATH = "/home/krishanu_2021cs19/ACL2023/CMEx/Results/TEXTX.csv"
    #SAVE_VIS_PATH
 
    GPU_N = 0
 