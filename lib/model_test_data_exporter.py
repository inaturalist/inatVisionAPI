import prison
import re
import asyncio
import aiohttp
import math


class ModelTestDataExporter:
    API_BASE_URL = "https://api.inaturalist.org/v2/observations"
    API_REQUEST_PER_PAGE = 200
    N_WORKERS = 3

    def __init__(self, export_path, max_results=5000, parameters={}, train_data_photo_ids=[]):
        self.export_path = export_path
        self.max_results = max_results
        self.parameters = parameters
        self.train_data_photo_ids = train_data_photo_ids
        self.rows_written = 0
        self.iterations_with_zero_results = 0
        self.used_observations = {}
        self.prepare_test_data_parameters()
        self.create_output_file()

    async def generate(self):
        async with aiohttp.ClientSession() as self.session:
            await self.generate_test_data()

    def create_output_file(self):
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
        self.output_file = open(self.export_path, "w")
        self.output_file.write(",".join(columns) + "\n")

    def prepare_test_data_parameters(self):
        api_parameters = {}
        api_parameters["quality_grade"] = "research,casual"
        api_parameters["rank"] = "species"
        api_parameters["photos"] = "true"
        api_parameters["geo"] = "true"
        api_parameters["identified"] = "true"
        api_parameters["per_page"] = ModelTestDataExporter.API_REQUEST_PER_PAGE
        api_parameters["order_by"] = "random"
        api_parameters["identifications"] = "most_agree"
        api_parameters["ttl"] = -1
        api_parameters.update(self.parameters)
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
        self.api_parameters = api_parameters

    async def worker_task(self):
        while not self.queue.empty():
            await self.queue.get()
            try:
                await self.process_api_response()
            finally:
                self.queue.task_done()

    async def generate_test_data(self):
        while not self.finished():
            await self.fetch_more_data()

    async def fetch_more_data(self):
        self.queue = asyncio.Queue()
        self.workers = [asyncio.create_task(self.worker_task())
                        for _ in range(ModelTestDataExporter.N_WORKERS)]
        min_pages_remaining = math.ceil(
            (self.max_results / ModelTestDataExporter.API_REQUEST_PER_PAGE)
        )
        print(f"Queueing {min_pages_remaining} workers")
        for i in range(min_pages_remaining):
            self.queue.put_nowait(i)
        await self.queue.join()
        for worker in self.workers:
            worker.cancel()

    def finished(self):
        return (self.rows_written >= self.max_results) or \
               (self.iterations_with_zero_results >= 5)

    async def process_api_response(self):
        if self.finished():
            return

        print(f"Fetching more results... {self.rows_written} so far")
        starting_rows_written = self.rows_written
        async with self.session.get(ModelTestDataExporter.API_BASE_URL,
                                    params=self.api_parameters) as response:
            json_object = await response.json()
            for row in json_object["results"]:
                self.process_api_response_row(row)
        if self.rows_written == starting_rows_written:
            self.iterations_with_zero_results += 1
            return

        self.iterations_with_zero_results = 0

    def process_api_response_row(self, row):
        if row["uuid"] in self.used_observations:
            return

        # must have a taxon and observed_on_details
        if not row["taxon"] or "observed_on_details" not in row \
           or not row["observed_on_details"] or "taxon" not in row \
           or "iconic_taxon_id" not in row["taxon"] or not row["taxon"]["iconic_taxon_id"]:
            self.used_observations[row["uuid"]] = True
            return

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
           or ("subject" in metric_counts and metric_counts["subject"] < 0) \
           or ("date" in metric_counts and metric_counts["date"] < 0) \
           or ("recent" in metric_counts and metric_counts["recent"] < 0):
            self.used_observations[row["uuid"]] = True
            return

        # check if any photos are included in the test data
        if self.photo_in_training_data(row) is True:
            self.used_observations[row["uuid"]] = True
            return
        if re.search(r"\.jpe?g", row["photos"][0]["url"]) is None:
            self.used_observations[row["uuid"]] = True
            return

        if row["quality_grade"] == "casual" \
           and not (row["community_taxon_id"] and row["community_taxon_id"] == row["taxon"]["id"]):
            self.used_observations[row["uuid"]] = True
            return

        self.used_observations[row["uuid"]] = True
        self.write_row(row)
        self.rows_written += 1

    def photo_in_training_data(self, row):
        for photo in row["photos"]:
            if photo["id"] in self.train_data_photo_ids:
                return True
        return False

    def write_row(self, row):
        if self.rows_written >= self.max_results:
            return
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
        self.output_file.write(",".join(map(str, columns_to_write)) + "\n")
