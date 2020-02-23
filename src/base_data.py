
class BaseData:
    def __init__(self):
        self.train = {'handle': None, 'indice': None, 'n_samples': None}
        self.val = {'handle': None, 'indice': None, 'n_samples': None}
        self.query = {'handle': None, 'indice': None, 'n_samples': None}
        self.gallery = {'handle': None, 'indice': None, 'n_samples': None}