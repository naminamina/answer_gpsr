import datetime
from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer("all-MiniLM-L6-v2")
model1 = SentenceTransformer("all-MiniLM-L6-v2")

import rospy

class AnswerQuestion:
    
    def __init__(self):
        self.date = datetime.datetime.now()
        self.tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        #質問と返答の辞書
        self.ANSWER_DICTIONARY = {
                    "the name of your team":"Our team's name is KIT HAPPY ROBOT",
                    "your name":"I am Happy Mimi",
                    "your team's country":"Our team's country is Japan",
                    "What day is today": self.date.strftime("%B%d"),
                    "What day is tomorrow":self.tomorrow.strftime("%B%d"),
                    "tell the day of the month":self.date.strftime('%B'),
                    "tell the day of the week":self.date.strftime('%A'),
                    "tell the date":self.date.strftime("%B%d"),
                    "what time is it":self.date.strftime("%H%M"),
                    "the time":self.date.strftime("%H%M")
                    }

    #user_questionと辞書のKEYのコサイン類似度から返答の決定
    def returnCosinsimilarity(self, sentences, user_question):
        embeddings = model.encode(sentences, convert_to_tensor=True)
        sentence = user_question
        print(sentence)
        
        embedding = model.encode(sentence, convert_to_tensor=True)
        socores = util.cos_sim(embedding, embeddings)
        return sentences[socores.argmax(1).item()]

    def returnAnswer(self, user_question) :
        sentences = []
        for i in self.ANSWER_DICTIONARY:
            sentences.append(i)

        result = self.returnCosinsimilarity(sentences, user_question)
        
        print(type(self.ANSWER_DICTIONARY[result]))
        return self.ANSWER_DICTIONARY[result]




def main(user_question):
    return AnswerQuestion().returnAnswer(user_question)

if __name__ == '__main__':
    print(AnswerQuestion().returnAnswer("country"))
