from Board.list import list


boardset = list.getboardset()
class Hexagon:

    def __init__ (self,amount):
        self.amount = amount
        self.boardset = boardset

    def tile(self):
        for i in range(self.amount):
            for j in range(6):
                self.list[j].append(i)

        return self.list
    

    def creategameboard(self):
        for i in range(len(18)):
            self.list[i] = [[i],[self.resource[i]],[self.probability[i]]]

    
    def setresource(self,resource):
        self.resource = resource
    

    def setprobability(self,probability):
        self.probability = probability

    def setsettlement(self,settlement):
        self.settlement = settlement

    def updateboard(self):
        