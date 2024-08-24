from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
import json
ref = json.load(open("./data/val_data.json", "r", encoding='utf-8'))
can = json.load(open("./output/result.json", "r", encoding='utf-8'))
BLEU1 = BLEU2 = BLEU3 = BLEU4 = GLEU = Meteor = 0
for i in range(len(can)):
    reference = ref['annotations'][i]['segment_caption']
    candidate = can[i]['caption']

    BLEU1 += sentence_bleu([word_tokenize(reference)], word_tokenize(candidate), weights=(1, 0, 0, 0))
    BLEU2 += sentence_bleu([word_tokenize(reference)], word_tokenize(candidate), weights=(0.5, 0.5, 0, 0))
    BLEU3 += sentence_bleu([word_tokenize(reference)], word_tokenize(candidate), weights=(0.33, 0.33, 0.33, 0))
    BLEU4 += sentence_bleu([word_tokenize(reference)], word_tokenize(candidate), weights=(0.25, 0.25, 0.25, 0.25))
    GLEU += sentence_gleu([word_tokenize(reference)], word_tokenize(candidate))
    Meteor += meteor_score([word_tokenize(reference)], word_tokenize(candidate))

print("BLEU1:"+ str(BLEU1/len(can)))
print("BLEU2:"+ str(BLEU2/len(can)))
print("BLEU3:"+ str(BLEU3/len(can)))
print("BLEU4:"+ str(BLEU4/len(can)))
print("GLEU:"+ str(GLEU/len(can)))
print("Meteor:"+ str(Meteor/len(can)))
