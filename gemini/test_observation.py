class TestObservation:

    def __init__(self, row):
        for key in row:
            setattr(self, key, row[key])

    def to_dict(self):
        return vars(self)