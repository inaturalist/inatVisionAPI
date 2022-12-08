class ModelScoring:

    @staticmethod
    def combine_vision_and_geo_scores(vision_scores, geo_scores):
        combined_scores = {}
        for arg in vision_scores:
            geo_score = geo_scores[arg] if arg in geo_scores else 0.0000000001
            combined_scores[arg] = vision_scores[arg] * geo_score
        return combined_scores
