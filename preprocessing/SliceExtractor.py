from deepbrain import Extractor
import warnings
warnings.filterwarnings("default")


class SliceExtractor:

    def __init__(self):
        # self.ext = extractor
        self.ext = Extractor()

    def get_prob(self, img):
        return self.ext.run(img)


extractor = SliceExtractor()
