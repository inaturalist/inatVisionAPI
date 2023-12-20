import pandas as pd
from datetime import datetime
from lib.model_test_data_exporter import ModelTestDataExporter


class ModelTestDataExportManager:
    def __init__(self, **args):
        self.cmd_args = args
        self.load_train_data_photo_ids()

    def load_train_data_photo_ids(self):
        if not self.cmd_args["exclude_train_photos_path"]:
            self.train_data_photo_ids = []
            return

        self.train_data_photo_ids = pd.concat(
            map(lambda x: pd.read_csv(x, usecols=["photo_id"]),
                self.cmd_args["exclude_train_photos_path"])
        ).drop_duplicates("photo_id").set_index("photo_id").sort_index().index

    def export_path(self, filename_addition):
        currentDatetime = datetime.now()
        timestamp = currentDatetime.strftime("%Y%m%d")
        export_path = f'test-obs-{timestamp}'
        if filename_addition:
            export_path += f'-{filename_addition}'
        if "filename_suffix" in self.cmd_args and self.cmd_args["filename_suffix"]:
            export_path += "-" + self.cmd_args["filename_suffix"]
        export_path += ".csv"
        return export_path

    async def generate_from_cmd_args(self):
        api_parameters = {}
        if self.cmd_args["place_id"]:
            api_parameters["place_id"] = self.cmd_args["place_id"]
        if self.cmd_args["taxon_id"]:
            api_parameters["taxon_id"] = self.cmd_args["taxon_id"]

        parameters_string = None
        if api_parameters:
            parameters_string = "-".join(map(lambda key: f'{key}-{api_parameters[key]}',
                                             api_parameters))
        export_path = self.export_path(parameters_string)
        exporter = ModelTestDataExporter(
            export_path=export_path,
            max_results=self.cmd_args["limit"],
            parameters=api_parameters,
            train_data_photo_ids=self.train_data_photo_ids
        )
        await exporter.generate()

    async def generate_standard_set(self):
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
        for key in files_to_generate:
            export_path = self.export_path(key)
            exporter = ModelTestDataExporter(
                export_path=export_path,
                max_results=self.cmd_args["limit"],
                parameters=files_to_generate[key],
                train_data_photo_ids=self.train_data_photo_ids
            )
            await exporter.generate()
