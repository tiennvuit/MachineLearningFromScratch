# Definition for base class 
class AlgorithmML():

    def __init__(self, name="ML Algorithm", data=None, args=None):
        
        self.name = name
        self.parameters = dict()


    def __str__(self) -> str:
        print(self.name)


    def learn(self):
        pass


    def evaluate(self):
        pass


    def predict(self):
        pass