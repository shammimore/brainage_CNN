from brainage.tools import crop_center


class ImageCropper():
    label = 'crop_center'

    def __init__(self):
        return 
    
    def transform(self, image_data, image_dimensions):
        return crop_center(image_data, image_dimensions)


