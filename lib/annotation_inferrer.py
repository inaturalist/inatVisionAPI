import json
import os

import numpy as np


class AnnotationInferrer:

    HEADS_FILE = "heads.npz"
    CONFIG_FILE = "head_config.json"
    LABEL_MAPS_FILE = "label_maps.json"
    THRESHOLDS_FILE = "thresholds.json"
    RULES_FILE = "controlled_term_rules.json"

    LIFE_TAXON_ID = 48460

    def __init__(self, model_dir, taxonomy):
        self.model_dir = model_dir
        self.taxonomy = taxonomy
        self.input_vector_cache = {}
        self.setup_config()
        self.setup_weights()
        self.setup_labels()
        self.setup_applicability()

    @staticmethod
    def bundle_is_present(model_dir):
        if not model_dir or not os.path.isdir(model_dir):
            return False
        return all(os.path.exists(os.path.join(model_dir, filename)) for filename in [
            AnnotationInferrer.HEADS_FILE,
            AnnotationInferrer.CONFIG_FILE,
            AnnotationInferrer.LABEL_MAPS_FILE,
            AnnotationInferrer.THRESHOLDS_FILE,
            AnnotationInferrer.RULES_FILE
        ])

    def setup_config(self):
        self.config = self.load_json(AnnotationInferrer.CONFIG_FILE)
        self.ce_attrs = self.config["ce_attrs"]
        self.bce_attrs = self.config["bce_attrs"]
        self.attrs = self.ce_attrs + self.bce_attrs
        self.run = self.config.get("run", "unknown")
        self.input_dim = self.config["input_dim"]
        self.embed_dim = self.config["embed_dim"]
        self.taxon_vector_dim = self.config["taxon_vector_dim"]

    def setup_weights(self):
        npz = np.load(os.path.join(self.model_dir, AnnotationInferrer.HEADS_FILE))
        self.weights = {
            attr: (
                npz[f"{attr}__w1"], npz[f"{attr}__b1"],
                npz[f"{attr}__w2"], npz[f"{attr}__b2"]
            ) for attr in self.attrs
        }
        for attr, (w1, _, w2, _) in self.weights.items():
            if w1.shape[0] != self.input_dim:
                raise ValueError(
                    f"{attr} head expects {w1.shape[0]}d input, head_config.json says "
                    f"{self.input_dim}d"
                )
            if w2.shape[1] != self.config["n_classes"][attr]:
                raise ValueError(f"{attr} head output width disagrees with head_config.json")

    def setup_labels(self):
        label_maps = self.load_json(AnnotationInferrer.LABEL_MAPS_FILE)
        self.values = {}
        for attr in self.ce_attrs:
            value_map = label_maps["ce"][attr]
            self.values[attr] = [v for v, _ in sorted(value_map.items(), key=lambda kv: kv[1])]
        for attr in self.bce_attrs:
            self.values[attr] = label_maps["bce"][attr]

        thresholds = self.load_json(AnnotationInferrer.THRESHOLDS_FILE)
        self.thresholds = {
            attr: np.asarray(thresholds[attr], dtype=np.float32) for attr in self.bce_attrs
        }
        for attr in self.attrs:
            if len(self.values[attr]) != self.config["n_classes"][attr]:
                raise ValueError(
                    f"{attr} has {len(self.values[attr])} labels but the head emits "
                    f"{self.config['n_classes'][attr]} values"
                )

    def setup_applicability(self):
        rules = self.load_json(AnnotationInferrer.RULES_FILE)
        rules_by_term_id = {term["id"]: term for term in rules}
        self.term_ids = self.config["taxon_vector_term_ids"]
        if len(self.term_ids) != self.taxon_vector_dim:
            raise ValueError(
                f"head_config.json lists {len(self.term_ids)} taxon vector terms but declares "
                f"taxon_vector_dim {self.taxon_vector_dim}"
            )
        missing = [term_id for term_id in self.term_ids if term_id not in rules_by_term_id]
        if missing:
            raise ValueError(
                f"{AnnotationInferrer.RULES_FILE} is missing terms the heads were trained on: "
                f"{missing}. Re-export the serving bundle rather than replacing the rules file."
            )

        self.term_rules = []
        for term_id in self.term_ids:
            taxa_rules = rules_by_term_id[term_id]["taxa_rules"]
            self.term_rules.append((
                frozenset(r["taxon_id"] for r in taxa_rules if not r["exception"]),
                frozenset(r["taxon_id"] for r in taxa_rules if r["exception"])
            ))

        column_index = {term_id: i for i, term_id in enumerate(self.term_ids)}
        self.attribute_columns = {
            attr: column_index[term_id]
            for attr, term_id in self.config["attribute_term_ids"].items()
        }
        self.value_columns = {
            attr: [column_index[term_id] for term_id in term_ids]
            for attr, term_ids in self.config["value_term_ids"].items()
        }

    def load_json(self, filename):
        with open(os.path.join(self.model_dir, filename)) as f:
            return json.load(f)

    def applicability_vector(self, ancestors):
        vector = np.zeros(self.taxon_vector_dim, dtype=np.float32)
        for index, (inclusions, exclusions) in enumerate(self.term_rules):
            if exclusions & ancestors:
                continue
            if not inclusions or inclusions & ancestors:
                vector[index] = 1.0
        return vector

    def input_ancestors(self, taxon_id, ancestor_ids=None):
        known = self.taxonomy.taxon_ancestors
        if taxon_id in known:
            return frozenset(known[taxon_id]) | {taxon_id}, "taxonomy"

        if ancestor_ids:
            rolled_up = frozenset(
                ancestor_id for ancestor_id in ancestor_ids if ancestor_id in known
            )
            if rolled_up:
                return rolled_up, "rolled_up"

        return frozenset([taxon_id]), "self_only"

    def mask_ancestors(self, taxon_id, ancestor_ids=None):
        ancestors = set(ancestor_ids) if ancestor_ids else set()
        if taxon_id in self.taxonomy.taxon_ancestors:
            ancestors.update(self.taxonomy.taxon_ancestors[taxon_id])
        if taxon_id is not None:
            ancestors.add(taxon_id)
        ancestors.add(AnnotationInferrer.LIFE_TAXON_ID)
        return frozenset(ancestors)

    def head_logits(self, x, attr):
        w1, b1, w2, b2 = self.weights[attr]
        hidden = np.maximum(x @ w1 + b1, 0.0)
        return hidden @ w2 + b2

    def predict(self, features, taxon_id, ancestor_ids=None, mask_by_applicability=True):
        embedding = np.asarray(features, dtype=np.float32).reshape(-1)
        if embedding.shape[0] != self.embed_dim:
            raise ValueError(
                f"Expected a {self.embed_dim}d feature vector, got {embedding.shape[0]}d. The "
                f"heads were trained on the CV model's penultimate layer — check that the vision "
                f"model matches the one the heads were trained against."
            )

        input_ancestors, ancestry_source = self.input_ancestors(taxon_id, ancestor_ids)
        taxon_vector = self.input_vector(input_ancestors, ancestry_source, taxon_id)
        x = np.concatenate([embedding, taxon_vector])

        off_distribution = (ancestry_source == "self_only")
        masking = mask_by_applicability and not off_distribution
        if masking:
            mask_vector = self.applicability_vector(
                self.mask_ancestors(taxon_id, ancestor_ids)
            )
        else:
            mask_vector = np.ones(self.taxon_vector_dim, dtype=np.float32)

        attributes = {}
        for attr in self.attrs:
            if not mask_vector[self.attribute_columns[attr]]:
                attributes[attr] = {"applicable": False}
                continue

            value_mask = mask_vector[self.value_columns[attr]] == 1.0
            logits = self.head_logits(x, attr)
            if attr in self.ce_attrs:
                attributes[attr] = self.ce_result(attr, logits, value_mask)
            else:
                attributes[attr] = self.bce_result(attr, logits, value_mask)

        return {
            "run": self.run,
            "taxon_id": taxon_id,
            "ancestry_source": ancestry_source,
            "off_distribution": off_distribution,
            "masked_by_applicability": masking,
            "attributes": attributes
        }

    def input_vector(self, ancestors, ancestry_source, taxon_id):
        cache_key = (taxon_id, ancestry_source, ancestors)
        if cache_key not in self.input_vector_cache:
            self.input_vector_cache[cache_key] = self.applicability_vector(ancestors)
        return self.input_vector_cache[cache_key]

    def ce_result(self, attr, logits, value_mask):
        if not value_mask.any():
            return {"applicable": False}

        masked = np.where(value_mask, logits, -np.inf)
        probabilities = AnnotationInferrer.softmax(masked)
        scores = {
            value: float(probabilities[i])
            for i, value in enumerate(self.values[attr]) if value_mask[i]
        }
        best = int(np.argmax(probabilities))
        return {
            "applicable": True,
            "prediction": self.values[attr][best],
            "score": float(probabilities[best]),
            "scores": scores
        }

    def bce_result(self, attr, logits, value_mask):
        probabilities = AnnotationInferrer.sigmoid(logits)
        over = probabilities >= self.thresholds[attr]
        scores = {}
        predictions = []
        for i, value in enumerate(self.values[attr]):
            if not value_mask[i]:
                continue
            scores[value] = float(probabilities[i])
            if over[i]:
                predictions.append(value)
        if not scores:
            return {"applicable": False}
        predictions.sort(key=lambda value: scores[value], reverse=True)
        return {
            "applicable": True,
            "predictions": predictions,
            "scores": scores
        }

    @staticmethod
    def softmax(logits):
        shifted = logits - np.max(logits)
        exponentiated = np.exp(shifted)
        return exponentiated / np.sum(exponentiated)

    @staticmethod
    def sigmoid(logits):
        return 1.0 / (1.0 + np.exp(-logits))
