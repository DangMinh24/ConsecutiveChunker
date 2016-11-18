import nltk
from nltk.corpus import conll2000
from naive import NaiveBayesModel

# class ConsecutiveChunker():


chunked_corpus=conll2000.chunked_sents("train.txt",chunk_types=["NP"],tagset="universal")
print(chunked_corpus)

iob_corpus=[nltk.chunk.tree2conlltags(tree) for tree in chunked_corpus]

def feature_extractor(sentence,i,history_pos,history_iob):
    feature_set={}
    feature_set["word"]=sentence[i]
    feature_set["pos"]=history_pos[i]
    if i==0:
        feature_set["previous_word"]="<START>"
        feature_set["previous_pos"]="<START_POS>"
        feature_set["previous_iob"]="None"
    else:
        feature_set["previous_word"]=sentence[i-1]
        feature_set["previous_pos"]=history_pos[i-1]
        feature_set["previous_iob"]=history_iob[i-1]
    return feature_set

train_data=[]

for sent in iob_corpus:
    sentence=[w for w,p,c in sent]
    history_pos=[p for w,p,c in sent]
    history_iob=[c for w,p,c in sent]

    for iter,(w,p,c) in enumerate(sent):
        train_data.append((feature_extractor(sentence,iter,history_pos,history_iob),c))


class ConsecutiveChunker(NaiveBayesModel):
    def tag(self,tagged_sentence):
        raw_sent=[w for w,p in tagged_sentence]
        history_pos=[p for w,p in tagged_sentence]
        history_iob=[]
        for iter,word in enumerate(raw_sent):
            tmp=[(feature_extractor(raw_sent,iter,history_pos,history_iob))]
            prediction=self.predict(tmp)
            history_iob.extend(prediction)
        return history_iob

model=ConsecutiveChunker()
model.train(train_data)

first_sent=iob_corpus[0]
print(first_sent)

first_sentence=[w for w,p,c in first_sent]
first_hist_pos=[w for w,p,c in first_sent]

golden_iob=[c for w,p,c in first_sent]

tagged_sentence=[(w,p) for w,p,c in first_sent]
prediction=model.tag(tagged_sentence)
print(prediction)
print(golden_iob)