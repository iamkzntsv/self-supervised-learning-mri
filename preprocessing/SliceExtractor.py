from preprocessing.extractor import extractor
from deepbrain import Extractor


class SliceExtractor:

    def __init__(self):
        # self.ext = extractor
        self.ext = Extractor()

    def get_prob(self, img):
        return self.ext(img)



extractor = SliceExtractor()
