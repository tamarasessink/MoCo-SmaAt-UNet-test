
class PercipationDataset:
    def __init__(self, traindir):
        self.traindir = traindir

    def __str__(self, traindir):
        for image in self.traindir:
            print(image.shape)



