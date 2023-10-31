import pandas as pd
import requests
import json
import prison
import re
from datetime import datetime


class ModelTestDataExporter:

    def __init__(self, **args):
        self.cmd_args = args
        self.load_train_data_photo_ids()

    def load_train_data_photo_ids(self, train_data_paths=[]):
        if not self.cmd_args["exclude_train_photos_path"]:
            self.train_data_photo_ids = []
            return
        self.train_data_photo_ids = pd.concat(
            map(lambda x: pd.read_csv(x, usecols=["photo_id"]),
                self.cmd_args["exclude_train_photos_path"])
        ).drop_duplicates("photo_id").set_index("photo_id").sort_index().index

    def generate_from_cmd_args(self):
        additional_parameters = {}
        if self.cmd_args["place_id"]:
            additional_parameters["place_id"] = self.cmd_args["place_id"]
        if self.cmd_args["taxon_id"]:
            additional_parameters["taxon_id"] = self.cmd_args["taxon_id"]
        currentDatetime = datetime.now()
        timestamp = currentDatetime.strftime("%Y%m%d")
        export_path = f'test-obs-{timestamp}'
        if additional_parameters:
            parameter_string = "-".join(map(lambda index: f'{index}-{additional_parameters[index]}',
                                            additional_parameters))
            export_path += "-" + parameter_string
        if "filename_suffix" in self.cmd_args and self.cmd_args["filename_suffix"]:
            export_path += "-" + self.cmd_args["filename_suffix"]
        export_path += ".csv"
        self.generate_test_data(export_path, self.cmd_args["limit"], additional_parameters)

    def export_test_data_parameters(self, additional_parameters={}):
        api_parameters = {}
        api_parameters["quality_grade"] = "research,casual"
        api_parameters["rank"] = "species"
        api_parameters["photos"] = "true"
        api_parameters["geo"] = "true"
        api_parameters["identified"] = "true"
        api_parameters["per_page"] = 200
        api_parameters["order_by"] = "random"
        api_parameters["identifications"] = "most_agree"
        api_parameters["ttl"] = -1
        api_parameters.update(additional_parameters)
        print(api_parameters)

        fields = {
            "quality_grade": True,
            "observed_on_details": {
                "date": True
            },
            "location": True,
            "photos": {
                "url": True
            },
            "community_taxon_id": True,
            "taxon": {
                "id": True,
                "ancestor_ids": True,
                "iconic_taxon_id": True
            },
            "quality_metrics": {
                "agree": True,
                "metric": True
            }
        }
        api_parameters["fields"] = prison.dumps(fields)
        return api_parameters

    def process_api_response(self, api_parameters, used_observations):  # noqa: C901
        response = requests.get("https://api.inaturalist.org/v2/observations",
                                params=api_parameters)
        json_object = response.json()
        useable_rows = []
        for row in json_object["results"]:
            if row["uuid"] in used_observations:
                continue

            # must have a taxon and observed_on_details
            if not row["taxon"] or "observed_on_details" not in row \
               or not row["observed_on_details"] or "taxon" not in row \
               or "iconic_taxon_id" not in row["taxon"] or not row["taxon"]["iconic_taxon_id"]:
                used_observations[row["uuid"]] = True
                continue

            # must pass quality metrics except wild
            metric_counts = {}
            for metric in row["quality_metrics"]:
                if metric["metric"] not in metric_counts:
                    metric_counts[metric["metric"]] = 0
                if metric["agree"]:
                    metric_counts[metric["metric"]] += 1
                else:
                    metric_counts[metric["metric"]] -= 1
            if ("location" in metric_counts and metric_counts["location"] < 0) \
               or ("evidence" in metric_counts and metric_counts["evidence"] < 0) \
               or ("date" in metric_counts and metric_counts["date"] < 0) \
               or ("recent" in metric_counts and metric_counts["recent"] < 0):
                used_observations[row["uuid"]] = True
                continue

            # check if any photos are included in the test data
            photo_in_training_data = False
            for photo in row["photos"]:
                if photo["id"] in self.train_data_photo_ids:
                    photo_in_training_data = True
                    break
            if photo_in_training_data is True:
                used_observations[row["uuid"]] = True
                continue
            if re.search(r"\.jpe?g", row["photos"][0]["url"]) is None:
                used_observations[row["uuid"]] = True
                continue

            if row["quality_grade"] == "casual" and not (row["community_taxon_id"] and row["community_taxon_id"] == row["taxon"]["id"]):
                used_observations[row["uuid"]] = True
                continue

            useable_rows.append(row)
            used_observations[row["uuid"]] = True
        return useable_rows

    def generate_test_data(self, filename, max_results=5000, additional_parameters={}):
        iterations_with_zero_results = 0
        rows_to_use = []
        used_observations = {}
        api_parameters = self.export_test_data_parameters(additional_parameters)

        while len(rows_to_use) < max_results and iterations_with_zero_results < 5:
            print(f'Fetching more results... {len(rows_to_use)} so far')
            useable_rows = self.process_api_response(api_parameters, used_observations)
            print(f'{len(useable_rows)} this batch')
            if not useable_rows:
                iterations_with_zero_results += 1
                continue
            iterations_with_zero_results = 0
            rows_to_use += useable_rows

        columns = [
            "observation_id",
            "observed_on",
            "iconic_taxon_id",
            "taxon_id",
            "taxon_ancestry",
            "lat",
            "lng",
            "photo_url"
        ]
        output_file = open(filename, "w")
        output_file.write(",".join(columns) + "\n")
        for row in rows_to_use[:max_results]:
            [latitude, longitude] = row["location"].split(",")
            columns_to_write = [
                row["uuid"],
                row["observed_on_details"]["date"],
                row["taxon"]["iconic_taxon_id"],
                row["taxon"]["id"],
                "/".join(map(str, row["taxon"]["ancestor_ids"])),
                latitude,
                longitude,
                row["photos"][0]["url"].replace("square", "medium").replace("http://", "https://")
            ]
            output_file.write(",".join(map(str, columns_to_write)) + "\n")

    def generate_standard_set(self):
        files_to_generate = {
            "global": {},
            "fungi": {"taxon_id": 47170},
            "insecta": {"taxon_id": 47158},
            "mammalia": {"taxon_id": 40151},
            "plantae": {"taxon_id": 47126},
            "actinopterygii": {"taxon_id": 47178},
            "reptilia": {"taxon_id": 26036},
            "amphibia": {"taxon_id": 20978},
            "arachnida": {"taxon_id": 47119},
            "aves": {"taxon_id": 3},
            "animalia": {"taxon_id": 1},
            "mollusca": {"taxon_id": 47115},
            "north-america": {"place_id": 97394},
            "south-america": {"place_id": 97389},
            "europe": {"place_id": 97391},
            "asia": {"place_id": 97395},
            "africa": {"place_id": 97392},
            "oceania": {"place_id": 97393}
        }

        currentDatetime = datetime.now()
        timestamp = currentDatetime.strftime("%Y%m%d")

        for key in files_to_generate:
            export_path = f'test-obs-{timestamp}-{key}'
            if "filename_suffix" in self.cmd_args and self.cmd_args["filename_suffix"]:
                export_path += "-" + self.cmd_args["filename_suffix"]
            export_path += ".csv"
            self.generate_test_data(export_path, self.cmd_args["limit"], files_to_generate[key])
