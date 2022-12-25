class PercipationDataset:
    def __init__(self, traindir):
        self.traindir = traindir

    def myfunc(self):
        for image in self.traindir:
          #(5734, 18, 288, 288)
          #in data frame stoppen
          print(image)