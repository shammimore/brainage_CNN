
class ImageNormalizer():
    label = 'normalize_image'

    def __init__(self):
        return 
    
    def transform(self, image_data, _):
        return image_data / image_data.mean()