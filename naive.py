import numpy as np
from nltk.corpus import brown
from nltk import defaultdict


class NaiveBayesModel():
    def __init__(self):
        self.set_attributes=[]
        self.labels=[]
        self.count_wl=dict()
        self.count_l=defaultdict(lambda :0)
        self.probability_wl=dict()
        self.probability_l=defaultdict(lambda :0)
        pass


    def train(self,corpus,alpha=0.1):
        self.set_attributes=list()
        for key in corpus[0][0].keys():
            self.set_attributes.append(key)
        self.labels=list(set([l for w,l in corpus]))

        self.count_wl=dict()
        for i in self.set_attributes:
            self.count_wl[i]=dict()
            for l in self.labels:
                self.count_wl[i][l]=defaultdict(lambda :0)


        self.probability_wl=dict()
        for i in self.set_attributes:
            self.probability_wl[i]=dict()
            for l in self.labels:
                self.probability_wl[i][l]=defaultdict(lambda :0)

        self.count_l=defaultdict(lambda :0)
        for fs,l in corpus:
            for key,val in fs.items():
                self.count_wl[key][l][val]+=1
            self.count_l[l]+=1

        for key in self.labels:
            self.probability_l[key]=float(self.count_l[key])/np.sum(list(self.count_l.values()))


    def predict(self,sent):

        ###------------------------
        # When input a whole new sentence, probability to appear a word that not in training corpus very high
        # Need to smoothing the whole count_wl
        #
        def smoothing(sent,count_wl,set_label,set_attributes,alpha=0.1):

            for fs in sent:
                for key,val in fs.items():
                    for l in set_label:
                        count_wl[key][l][val]+=0
            for attribute in set_attributes:
                for label in set_label:
                    for key,val in count_wl[attribute][label].items():
                        count_wl[attribute][label][key]+=alpha
            return count_wl

        self.count_wl=smoothing(sent,self.count_wl,self.labels,self.set_attributes)


        def probability(count_wl,count_l,probability_wl,alpha=0.1):

            for attribute in count_wl.keys():
                for label in count_wl[attribute].keys():
                    for value_name in count_wl[attribute][label].keys():
                        probability_wl[attribute][label][value_name]=float(count_wl[attribute][label][value_name])/(count_l[label]      )
            return probability_wl

        self.probability_wl=probability(self.count_wl,self.count_l,self.probability_wl)

        prediction=[]
        for fs in sent:
            result=[]
            result_label=[]
            for label  in self.labels:
                score=0
                for feature,val_name in fs.items():
                    score_tmp=self.probability_wl[feature][label][val_name]*self.probability_l[label]
                    score+=score_tmp
                result.append(score)
                result_label.append(label)
            prediction.append(result_label[np.argmax(result)])
        return prediction

    def predict_many(self,sents):
        predictions=[]
        for sent in sents:
            prediction=self.predict(sent)
            predictions.append(prediction)
        return predictions




