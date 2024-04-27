import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu



ref = "/home/krishanu_2021cs19/ACL2023/CMEx/VisualExplanations"
target = "/home/krishanu_2021cs19/ACL2023/CMEx/Predicted"


#out_test = open("/home/krishanu_2021cs19/ACL2023/VLCybebully/Dataset/Visual_Textual_Data/test_id.pkl","rb")
#test_id = pickle.load(out_test)

test_data = pd.read_csv("/home/krishanu_2021cs19/ACL2023/VLCybebully/TextualExplanations/df_test.tsv",sep="\t")

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)


def segmentation_metrics(ref,target):
    dice = []
    js = []
    miou = []

    for i in range(len(test_data)):
    
        try:
    
            ref_path = os.path.join(ref,str(test_data["pid"][i])+".npy")
            target_path = os.path.join(target,test_data["pid"][i]+".npy")

            gt = np.load(ref_path).reshape(-1)
            seg = np.load(target_path).reshape(-1)

            dice.append(np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt)))
            js.append(jaccard_score(gt,seg))
            miou.append(compute_iou(seg,gt))
        
        except Exception as e:
            print(e)
            
    return dice,js,miou


   



dice,js,miou = segmentation_metrics(ref,target)

#print(dice)

print("dice coefficient",sum(dice)/len(dice))
print("jaccard_similarity_score",sum(js)/len(js))
print("miou score",sum(miou)/len(miou))

 
 

path = "/home/krishanu_2021cs19/ACL2023/CMEx/Results/TEXTX.csv"

df = pd.read_csv(path)

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)



rouge_1 = []
rouge_2 = []
rouge_l = []

bleu_1 = []
bleu_2 = []
bleu_3 = []
bleu_4 = []


for i in range(len(df)):

    scores = scorer.score(str(df["actual_explanation"][i]),str(df["predicted_explanation"][i]))
    
    rouge_1.append(scores["rouge1"][2])
    rouge_2.append(scores["rouge2"][2])
    rouge_l.append(scores["rougeL"][2])
    
    bleu_1.append(sentence_bleu([str(df["actual_explanation"][i])],str(df["predicted_explanation"][i]),weights=(1, 0, 0, 0)))
    bleu_2.append(sentence_bleu([str(df["actual_explanation"][i])],str(df["predicted_explanation"][i]),weights=(0, 1, 0, 0)))
    bleu_3.append(sentence_bleu([str(df["actual_explanation"][i])],str(df["predicted_explanation"][i]),weights=(0, 0, 1, 0)))
    bleu_4.append(sentence_bleu([str(df["actual_explanation"][i])],str(df["predicted_explanation"][i]),weights=(0, 0, 0, 1)))
    
    
print("rouge 1",sum(rouge_1)/len(rouge_1))
print("rouge 2",sum(rouge_2)/len(rouge_2))
print("rouge L",sum(rouge_l)/len(rouge_l))


print("bleu 1",sum(bleu_1)/len(bleu_1))
print("bleu 2",sum(bleu_2)/len(bleu_2))
print("bleu 3",sum(bleu_3)/len(bleu_3))
print("bleu 4",sum(bleu_4)/len(bleu_4))
        
