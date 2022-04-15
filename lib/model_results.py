class ModelResults:

    def __init__(self, vision_results, geo_results, taxonomy):
        self.taxonomy = taxonomy
        self.vision_results = vision_results
        self.geo_results = geo_results
        # common_ancestor is currently being used as a first-pass filter to remove
        # the least likely results and reduce the number of taxa whose scores to combine.
        # NOTE: This may not be helpful and needs testing for accuracy and processing time
        self.common_ancestor_threshold = 0.9
        self.common_ancestor_rank_level_threshold = 50
        # fine_common_ancestor is currently being used to return as a high-confidence
        # non-leaf taxon that may get presented to a user
        self.fine_common_ancestor_threshold = 0.85
        self.fine_common_ancestor_rank_level_threshold = 20
        # vision scores are raw unnormalized scores from the vision model
        # geo scores are raw unnormalized scores from the vision model
        # combined scores are the unnormalized product of vision and geo scores
        # combined_agg scores are the unnormalized sum of combined scores the descendants of a taxon
        self.scores = {
            "vision": {},
            "geo": {},
            "combined": {},
            "combined_agg": {}
        }

    def aggregate_scores(self):
        self.ancestor_scores = {}
        self.vision_sum_scores = 0
        # loop through all vision results, calculating the sum of vision scores for each ancestor
        for arg in self.vision_results:
            taxon = self.taxonomy.taxa[arg]
            self.vision_sum_scores += self.vision_results[arg]
            # add the score of this leaf result to all of its ancestors
            for ancestor in taxon.ancestors:
                if ancestor not in self.ancestor_scores:
                    self.ancestor_scores[ancestor] = 0
                self.ancestor_scores[ancestor] += self.vision_results[arg]

        # using only the vision results, calculate a highly-likely visual common ancestor
        # that is no narrower than self.common_ancestor_rank_level_threshold (currently Class).
        # Taxa outside the highly-likely visual common ancestor will be ignored
        # NOTE: This may not be helpful and needs testing for accuracy and processing time
        self.common_ancestor = self.calculate_common_ancestor(
            self.ancestor_scores, self.vision_sum_scores, self.common_ancestor_threshold,
            self.common_ancestor_rank_level_threshold)

        # loop through all taxa and combine geo and vision scores, calculating
        # aggregate scores for non-leaf taxa as well
        self.aggregate_scores_recursive()

        # after combining vision and geo scores, look for a potentially more-specific
        # common ancestor using the combined scores and different thresholds.
        # 0 represents the root taxon, so the combined aggretate score for 0
        # represents the sum of all combined scores of all leaves
        sum_of_all_combined_scores = self.scores["combined_agg"][0]
        self.fine_common_ancestor = self.calculate_common_ancestor(
            self.scores["combined_agg"], sum_of_all_combined_scores,
            self.fine_common_ancestor_threshold, self.fine_common_ancestor_rank_level_threshold)

    # given a set of scores, the sum of those scores (so we only need to calculate it once),
    # a score threshold, a rank_level threshold, and optionall a taxon (if none is given it starts
    # at the root of the taxonomy), resursively find the most specific node that is above
    # the specified thresholds
    def calculate_common_ancestor(self, ancestor_scores, sum_scores, score_threshold,
                                  rank_level_threshold, taxon=None):
        common_ancestor = taxon
        taxon_id = 0 if taxon is None else taxon.id
        # sort children from most- to least-likely
        for child_id in sorted(
            self.taxonomy.taxon_children[taxon_id],
            key=lambda x: (ancestor_scores[x] if x in ancestor_scores else 0),
                reverse=True):
            # the child has no scores. This could be the result of pruning scores
            # earlier on based on iconic_taxon. If there is no score, skip this branch
            if child_id not in ancestor_scores:
                break
            # if the ratio of this score to the sum of all scores is below the
            # score_threshold, then this taxon and its whole branch can be skipped
            if (ancestor_scores[child_id] / sum_scores) < score_threshold:
                break
            child_taxon = self.taxonomy.taxa[child_id]
            # if this taxon is below the rank_level_threshold, this branch can be skipped
            if child_taxon.rank_level < rank_level_threshold:
                continue
            # this is a leaf, so return it
            if child_id not in self.taxonomy.taxon_children:
                return child_taxon
            return self.calculate_common_ancestor(ancestor_scores, sum_scores, score_threshold,
                                                  rank_level_threshold, child_taxon)
        return common_ancestor

    # takes a taxonID of the branch to score, and an indication if the branch is
    # already known to be within the common ancestor branch
    def aggregate_scores_recursive(self, taxon_id=0, in_common_ancestor=False):
        vision_score = 0
        geo_score = 0
        combined_agg_score = 0
        # loop through all children of this iteration's taxon, or root taxon
        for child_id in self.taxonomy.taxon_children[taxon_id]:
            is_common_ancestor = False
            # if there is a common ancestor, and this taxon is not yet known to be in it
            if self.common_ancestor and not in_common_ancestor:
                if child_id == self.common_ancestor.id:
                    # keep track that this taxon is the common ancestor, and resursive calls from
                    # this node down are also within the common ancestor
                    is_common_ancestor = True
                elif child_id not in self.common_ancestor.ancestors:
                    # skip taxa that are not in the common ancestor branch
                    continue
            # this taxon has children in the model
            if child_id in self.taxonomy.taxon_children:
                self.aggregate_scores_recursive(child_id, in_common_ancestor or is_common_ancestor)
            else:
                # this is a leaf taxon in the model
                # record the vision and geo scores, using very low default scores for missing values
                if child_id in self.vision_results:
                    child_vision_score = self.vision_results[child_id]
                else:
                    child_vision_score = 0.00000001
                if child_id in self.geo_results:
                    child_geo_score = self.geo_results[child_id]
                else:
                    child_geo_score = 0.00000001
                self.scores["vision"][child_id] = child_vision_score
                self.scores["geo"][child_id] = child_geo_score
                # simple muliplication of vision and geo score to get a combined score
                self.scores["combined"][child_id] = child_vision_score * child_geo_score
                # also keeping track of scores aggregated up the tree. Since this is a leaf node,
                # the aggregate branch score is equal to the combined score
                self.scores["combined_agg"][child_id] = self.scores["combined"][child_id]

            child_vision_score = self.scores["vision"][child_id]
            child_geo_score = self.scores["geo"][child_id]
            child_combined_agg_score = self.scores["combined_agg"][child_id]

            # vision scores can just be summed as they'll add up to 1
            vision_score += child_vision_score
            # all maintain a sum of the combined scores in the branch. This will not add
            # up to 1 and can be a wide range of values. Useful when compared to the sum
            # of the combined scores for the entire tree
            combined_agg_score += child_combined_agg_score

            # geo scores do not add up to 1, so have the geo score of a
            # taxon be the max of the scores of its children
            if child_geo_score > geo_score:
                geo_score = child_geo_score
        # scores have been calculated and summed for all this taxon's descendants,
        # so reccord the final scores for this branch
        self.scores["vision"][taxon_id] = vision_score
        self.scores["geo"][taxon_id] = geo_score
        self.scores["combined_agg"][taxon_id] = combined_agg_score

    # prints to the console a tree prepresenting the most likely taxa and their
    # aggregate combined score ratio. e.g. if all combined scores add up to 0.5
    # and a taxon has a combined score of 0.1, its combined score ratio will be 20%, or 0.2
    def print(self, taxon_id=0, ancestor_prefix=""):
        children = self.taxonomy.taxon_children[taxon_id]
        # 0 represents the root taxon, so the combined aggretate score for 0
        # represents the sum of all combined scores of all leaves
        sum_of_all_commbined_scores = self.scores["combined_agg"][0]
        # ignore children whose combined score ration is less than 0.01
        scored_children = list(filter(lambda x: x in self.scores["combined_agg"] and (
            (self.scores["combined_agg"][x] / sum_of_all_commbined_scores) >= 0.01), children))
        # sort children by score from most- to least-likely
        scored_children = sorted(scored_children, key=lambda x: self.scores["combined_agg"][x],
                                 reverse=True)

        index = 0
        for child_id in scored_children:
            # some logic for visual tree indicators when printing
            last_in_branch = (index == len(scored_children) - 1)
            index += 1
            icon = "└──" if last_in_branch else "├──"
            prefixIcon = "   " if last_in_branch else "│   "
            taxon = self.taxonomy.taxa[child_id]
            # print the taxon with its combined score ratio
            combined_score_ratio = self.scores["combined_agg"][child_id] / self.scores["combined_agg"][0]
            print(f'{ancestor_prefix}{icon}{taxon.name} :: {combined_score_ratio:.10f}')
            # recursively repeat for descendants
            if child_id in self.taxonomy.taxon_children:
                self.print(child_id, f'{ancestor_prefix}{prefixIcon}')
