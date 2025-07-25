# 多模态输入


class Multimodal(object):
    def __init__(self, text, image):
        self.text = text
        self.image = image
    def __repr__(self):
        return 'Multimodal(text={}, image={})'.format(self.text, self.image)