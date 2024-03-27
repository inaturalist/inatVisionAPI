# TestObservation:
#   observation_id
#   observed_on
#   iconic_taxon_id
#   taxon_id
#   taxon_ancestry
#   lat
#   lng
#   photo_url


class TestObservation:

    def __init__(self, row):
        row["taxon_ancestry"] = row["taxon_ancestry"].split("/")
        row["taxon_ancestry"] = list(map(int, row["taxon_ancestry"]))
        # remove life
        row["taxon_ancestry"].pop(0)
        for key in row:
            setattr(self, key, row[key])
        self.inferrer_results = None
        self.summarized_results = {}
