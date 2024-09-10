import os
import hashlib
import magic
import time
import json
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import aiofiles
import aiofiles.os
import re
import traceback
import requests
from datetime import datetime
from test_observation import TestObservation

class GeminiEvalutation:

    TAXA_API_URL = "https://api.inaturalist.org/v2/taxa/"

    def __init__(self, **args):
        self.cmd_args = args
        currentDatetime = datetime.now()
        self.start_timestamp = currentDatetime.strftime("%Y%m%d")
   
    def export_path(self, path, file):
        label = self.cmd_args["label"]
        export_path = f"{label}-{file}"
        return os.path.join(self.cmd_args["data_dir"], export_path)
        
    async def run_async(self):
        if self.cmd_args["data_dir"]:
            for file in sorted(os.listdir(self.cmd_args["data_dir"])):
                exported_data_filename_match = re.search(r"test-results-[0-9]{8}-(.*).csv", file)
                if exported_data_filename_match is None:
                    continue
                path = os.path.join(self.cmd_args["data_dir"], file)
                print(f"\nProcessing {file}")
                await self.evaluate_observations_at_path(path)
                observations_export_path = self.export_path(path, file)
                self.display_and_save_results(observations_export_path)

    async def evaluate_observations_at_path(self, path):
        N_WORKERS = 5
        self.limit = self.cmd_args["limit"] or 100
        self.start_time = time.time()
        self.queued_counter = 0
        self.processed_counter = 0
        self.test_observations = {}

        async with aiohttp.ClientSession() as self.session:
            self.queue = asyncio.Queue()
            self.workers = [
                asyncio.create_task(self.worker_task()) for _ in range(N_WORKERS)
            ]
            df = pd.read_csv(
                path,
                dtype={
                    "observation_id": str,
                    "taxon_id": int,
                    "taxon_ancestry": str,
                    "gemini_response": str,
                    "gemini_error": str
                }
            )
            df = df.drop(df.columns[0], axis=1)
            for index, observation in df.iterrows():
                obs = TestObservation(observation.to_dict())
                if obs.gemini_error == "True":
                    continue
                self.test_observations[obs.observation_id] = obs
                self.queue.put_nowait(obs.observation_id)

            # processes the queue
            await self.queue.join()
            # stop the workers
            for worker in self.workers:
                worker.cancel()

    async def worker_task(self):
        while not self.queue.empty():
            observation_id = await self.queue.get()
            try:
                if self.processed_counter >= self.limit:
                    continue
                observation = self.test_observations[observation_id]
                await self.test_observation_async(observation)
                self.processed_counter += 1
                self.report_progress()

            except Exception as err:
                print(f"\nObservation: {observation_id} failed")
                print(traceback.format_exc())
                print(err)

            finally:
                self.queue.task_done()
    
    def string_raw_comparison(self, string1, string2):
        return string1.lower().replace(" ", "") == string2.lower().replace(" ", "")

    async def test_observation_async(self, observation):
        observation.clean_gemini_name = ''.join(char for char in observation.gemini_response if char.isalpha() or char.isspace() or char == '-')
        print(observation.clean_gemini_name)

        self.evaluate_taxon_name(observation)

        if not observation.evaluation_status:
            clean_gemini_name = observation.gemini_response.replace("*", "")
            gbif_response = requests.get("https://api.gbif.org/v1/parser/name?name="+clean_gemini_name)
            if gbif_response.status_code == 200:
                gbif_data = gbif_response.json()
                if gbif_data[0]:
                    observation.clean_gemini_name = gbif_data[0]["canonicalName"] 
                    observation.evaluate_using_gbif = True
                    print(observation.clean_gemini_name + " (from GBIF)")
                    
                    self.evaluate_taxon_name(observation)

        observation.matching_active = False
        observation.matching_synonym = False
        
        if observation.evaluation_status:
            if observation.evaluation_is_active and observation.evaluation_taxon_id == observation.taxon_id:
                observation.matching_active = True
            elif not observation.evaluation_is_active and observation.taxon_id in observation.evaluation_synonymous_taxon_ids:
                observation.matching_synonym = True

        observation.matching = observation.matching_active or observation.matching_synonym
        observation.matching_int = int(observation.matching)

    def evaluate_taxon_name(self, observation):
        original_taxa_url = self.TAXA_API_URL + str(observation.taxon_id) + "?fields=name,is_active,current_synonymous_taxon_ids"
        original_response = requests.get(original_taxa_url)
        if original_response.status_code == 200:
            data = original_response.json()
            observation.original_name = data["results"][0]["name"] 
            observation.original_is_active = data["results"][0]["is_active"] 
            observation.original_is_synonymous = data["results"][0]["current_synonymous_taxon_ids"] is not None

        observation.evaluation_status = False
        
        matching_active_taxa_url = self.TAXA_API_URL + "autocomplete?q=" + observation.clean_gemini_name + "&is_active=true&fields=id,name,is_active,current_synonymous_taxon_ids,rank"
        matching_active_response = requests.get(matching_active_taxa_url)

        if matching_active_response.status_code == 200:
            active_data = matching_active_response.json() 
        else:
            observation.evaluation_status = False
            observation.evaluation_error = "Error when calling iNat taxa suggest endpoint (active)"
            return

        for result in active_data["results"]:
            if self.string_raw_comparison(observation.clean_gemini_name, result["name"]):
                observation.evaluation_status = True
                observation.evaluation_name = result["name"]
                observation.evaluation_taxon_id = result["id"]
                observation.evaluation_rank = result["rank"]
                observation.evaluation_is_active = result["is_active"]
                break

        if not observation.evaluation_status:
            matching_inactive_taxa_url = self.TAXA_API_URL + "autocomplete?q=" + observation.clean_gemini_name + "&is_active=false&fields=id,name,is_active,current_synonymous_taxon_ids,rank"
            matching_inactive_response = requests.get(matching_inactive_taxa_url)

            if matching_inactive_response.status_code == 200:
                inactive_data = matching_inactive_response.json()   
            else:
                observation.evaluation_status = False
                observation.evaluation_error = "Error when calling iNat taxa suggest endpoint (inactive)"
                return

            for result in inactive_data["results"]:
                if self.string_raw_comparison(observation.clean_gemini_name, result["name"]):
                    observation.evaluation_status = True
                    observation.evaluation_name = result["name"]
                    observation.evaluation_taxon_id = result["id"]
                    observation.evaluation_rank = result["rank"]
                    observation.evaluation_is_active = result["is_active"]
                    observation.evaluation_synonymous_taxon_ids = result["current_synonymous_taxon_ids"]
                    break
    
    def display_and_save_results(self, observations_export_path):
        test_observations_data = [obs.to_dict() for obs in self.test_observations.values()]
        test_observations_df = pd.DataFrame(test_observations_data)
        test_observations_df.to_csv(observations_export_path)

    def report_progress(self):
        if self.processed_counter % 10 == 0:
            total_time = round(time.time() - self.start_time, 2)
            remaining_time = round((
                self.limit - self.processed_counter
            ) / (self.processed_counter / total_time), 2)
            rate = round(self.processed_counter / total_time, 2)
            print(
                f"Processed {self.processed_counter} in {total_time} sec  \t"
                f"{rate}/sec  \t"
                f"estimated {remaining_time} sec remaining\t"
            )
