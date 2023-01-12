class PercipationDataset:
    def __init__(self, traindir, transform=None):
        self.traindir = traindir
        self.transform = transform

    def __getitem__(self):

        if self.transform:
            for image in self.traindir:
                # (5734, 18, 288, 288)
                # in data frame stoppen
                sample = self.transform(sample)
        else:
            for image in self.traindir:
                # (5734, 18, 288, 288)
                # in data frame stoppen
                sample = image

        return sample

    def myfunc(self):
        for image in self.traindir:
            # (5734, 18, 288, 288)
            # in data frame stoppen
            print(image)