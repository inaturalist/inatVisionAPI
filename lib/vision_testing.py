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
from datetime import datetime
from PIL import Image
from lib.test_observation import TestObservation
from lib.inat_inferrer import InatInferrer


class VisionTesting:

    def __init__(self, config, **args):
        self.cmd_args = args
        self.inferrers = {}
        self.upload_folder = "static/"
        currentDatetime = datetime.now()
        self.start_timestamp = currentDatetime.strftime("%Y%m%d")
        self.set_run_hash(config)

        print("Models:")
        for inferrer_index, model_config in enumerate(config["models"]):
            print(json.dumps(model_config, indent=4))
            model_name = model_config["name"] if "name" in model_config \
                else f"Model {inferrer_index}"
            model_config["name"] = model_name
            self.inferrers[inferrer_index] = InatInferrer(model_config)

    def set_run_hash(self, config):
        run_hash_dict = dict(config)
        run_hash_dict.update(self.cmd_args)
        dhash = hashlib.md5()
        encoded = json.dumps(run_hash_dict, sort_keys=True).encode()
        dhash.update(encoded)
        self.run_hash = dhash.hexdigest()

    def export_path(self, filename_addition=None, label=None):
        if label is None:
            label = self.cmd_args["label"]
        export_path = f"test-results-{self.start_timestamp}-{label}-{self.run_hash}"
        if filename_addition:
            export_path += f"-{filename_addition}"
        export_path += ".csv"
        if self.cmd_args["data_dir"]:
            export_path = os.path.join(self.cmd_args["data_dir"], export_path)
        return export_path

    async def run_async(self):
        if self.cmd_args["data_dir"]:
            for file in sorted(os.listdir(self.cmd_args["data_dir"])):
                exported_data_filename_match = re.search(r"test-obs-[0-9]{8}-(.*).csv", file)
                if exported_data_filename_match is None:
                    continue
                label = exported_data_filename_match.group(1)
                path = os.path.join(self.cmd_args["data_dir"], file)
                print(f"\nProcessing {file}")
                await self.test_observations_at_path(path, label)
                self.display_and_save_results(label)
        else:
            print(f"\nProcessing {self.cmd_args['path']}")
            await self.test_observations_at_path(self.cmd_args["path"], self.cmd_args["label"])
            self.display_and_save_results(self.cmd_args["label"])

    async def test_observations_at_path(self, path, label):
        N_WORKERS = 5
        self.limit = self.cmd_args["limit"] or 100
        target_observation_id = self.cmd_args["observation_id"]
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
                    "iconic_taxon_id": float,
                    "taxon_id": int,
                    "lat": float,
                    "lng": float
                }
            )
            for index, observation in df.iterrows():
                if target_observation_id and observation.observation_id != target_observation_id:
                    continue
                obs = TestObservation(observation.to_dict())
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
                if observation.inferrer_results is None:
                    continue
                self.processed_counter += 1
                self.report_progress()

            except Exception as err:
                print(f"\nObservation: {observation_id} failed")
                print(traceback.format_exc())
                print(err)

            finally:
                self.queue.task_done()

    def display_and_save_results(self, label):
        scored_observations = list(filter(
            lambda observation: len(observation.summarized_results) > 0,
            self.test_observations.values()
        ))
        if len(scored_observations) == 0:
            return

        # extract a summary score for each observation for each inferrer and scoring method
        keys = scored_observations[0].summarized_results[0].keys()
        all_obs_scores = []
        for obs in scored_observations:
            for inferrer_index, inferrer in self.inferrers.items():
                for method in keys:
                    scores = {
                        "inferrer_name": inferrer.config["name"],
                        "label": label,
                        "method": method,
                        "uuid": obs.observation_id
                    }
                    scores.update(obs.summarized_results[inferrer_index][method])
                    all_obs_scores.append(scores)
        pd.set_option("display.max_rows", None)
        all_obs_scores_df = pd.DataFrame(all_obs_scores)
        all_obs_scores_df["run_label"] = all_obs_scores_df[
            ["inferrer_name", "label", "method"]
        ].agg("-".join, axis=1)
        observations_export_path = self.export_path("observations", label=label)
        all_obs_scores_df.to_csv(observations_export_path)

        # generate a summary score of all observations for each inferrer and scoring method
        grouped_stats = all_obs_scores_df.groupby("run_label").agg(
            inferrer_name=("inferrer_name", "max"),
            label=("label", "max"),
            method=("method", "max"),
            count=("label", "count"),
        )
        aggs = []
        aggs.append(all_obs_scores_df.query("matching_index == 0").groupby("run_label").agg(
            top1=("label", "count"),
        ))
        aggs.append(all_obs_scores_df.query("matching_index < 5").groupby("run_label").agg(
            top5=("label", "count"),
        ))
        aggs.append(all_obs_scores_df.query("matching_index < 10").groupby("run_label").agg(
            top10=("label", "count"),
        ))
        aggs.append(all_obs_scores_df.query("matching_index.isna()").groupby("run_label").agg(
            notIn=("label", "count"),
        ))
        aggs.append(all_obs_scores_df.query("common_ancestor_present == 1").groupby(
            "run_label"
        ).agg(
            withCA=("label", "count"),
        ))
        aggs.append(all_obs_scores_df.query("common_ancestor_accurate == 1").groupby(
            "run_label"
        ).agg(
            withRightCA=("label", "count"),
        ))
        aggs.append(all_obs_scores_df.query("common_ancestor_present == 1").groupby(
            "run_label"
        ).agg(
            CARankLevel=("common_ancestor_rank_level", "mean"),
        ))
        for agg in aggs:
            grouped_stats = grouped_stats.merge(
                agg, how="left", left_on="run_label", right_on="run_label"
            )
        grouped_stats["top1%"] = round((grouped_stats["top1"] / grouped_stats["count"]) * 100, 2)
        grouped_stats["top5%"] = round((grouped_stats["top5"] / grouped_stats["count"]) * 100, 2)
        grouped_stats["top10%"] = round((grouped_stats["top10"] / grouped_stats["count"]) * 100, 2)
        grouped_stats["notIn%"] = round((grouped_stats["notIn"] / grouped_stats["count"]) * 100, 2)
        grouped_stats["withCA%"] = round((
            grouped_stats["withCA"] / grouped_stats["count"]
        ) * 100, 2)
        grouped_stats["withRightCA%"] = round((
            grouped_stats["withRightCA"] / grouped_stats["count"]
        ) * 100, 2)
        grouped_stats["withRightCAWhenPresent%"] = round((
            grouped_stats["withRightCA"] / grouped_stats["withCA"]
        ) * 100, 2)

        agg_stats = all_obs_scores_df.groupby("run_label").agg(
            average_results_count=("results_count", "mean"),
            common_ancestor_pool_size=("common_ancestor_pool_size", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean")
        )
        grouped_stats = grouped_stats.merge(
            agg_stats, how="left", left_on="run_label", right_on="run_label"
        )
        grouped_stats["average_results_count"] = grouped_stats["average_results_count"].round(4)
        grouped_stats["common_ancestor_pool_size"] = grouped_stats[
            "common_ancestor_pool_size"
        ].round(4)
        grouped_stats["precision"] = grouped_stats["precision"].round(4)
        grouped_stats["recall"] = grouped_stats["recall"].round(4)
        grouped_stats["f1"] = grouped_stats["f1"].round(4)
        grouped_stats = grouped_stats.sort_index(ascending=False)
        print(grouped_stats[grouped_stats.columns.difference(["inferrer_name", "label", "method"])])

        grouped_stats_export_path = self.export_path("summary", label=label)
        grouped_stats.to_csv(grouped_stats_export_path)

    async def test_observation_async(self, observation):
        cache_path = await self.download_photo_async(observation.photo_url)

        # due to asynchronous processing, the requested limit of observations to test
        # has been reached, so do not test this observation. The rest of this method
        # will be processed synchronously, so no need to check this again this method
        if self.processed_counter >= self.limit:
            return

        if cache_path is None \
           or not os.path.exists(cache_path) \
           or observation.lat == "" \
           or observation.lng == "":
            return

        iconic_taxon_id = None
        if observation.iconic_taxon_id != "" and self.cmd_args["filter_iconic"] is not False:
            iconic_taxon_id = observation.iconic_taxon_id

        inferrer_results = {}
        summarized_results = {}
        for inferrer_index, inferrer in self.inferrers.items():
            lat = None
            lng = None
            filter_taxon = inferrer.lookup_taxon(iconic_taxon_id)
            if inferrer.geo_elevation_model and self.cmd_args["geo"]:
                lat = observation.lat
                lng = observation.lng
            try:
                # traditional leaf combined scores, vision * geo
                leaf_scores = inferrer.predictions_for_image(
                    cache_path, lat, lng, filter_taxon
                )
                # save some high-level data like top 100 scores, common ancestor
                inferrer_results[inferrer_index] = self.inferrer_results(
                    inferrer, observation, leaf_scores
                )
                # summarize that high-level data further into the metrics we ultimately want
                summaries = {}
                for summary_index, summary in inferrer_results[inferrer_index].items():
                    summaries[summary_index] = self.summarize_result_subset(
                        inferrer, observation, summary, summary_index)
                    # the "cutoff" variant limits the top100 retults into the subset users might
                    # actually be presented. That means removing results with a score less than
                    # 0.001 times the top score, and using up to 10 of those. This will give values
                    # for precision, recall, f1 relative to user experience rather than all results
                    summaries[f"{summary_index}-cutoff"] = self.summarize_result_subset(
                        inferrer, observation, summary, summary_index, cutoff=True
                    )
                summarized_results[inferrer_index] = summaries

            except Exception as e:
                print(f"Error scoring observation {observation.observation_id}")
                print(e)
                print(traceback.format_exc())
                return

        # record the results and summaries with the observation only after all have
        # finished without exception. This ensures we don't store partial results if some
        # inferrers fail, and we can later filter our observations without results
        observation.inferrer_results = inferrer_results
        observation.summarized_results = summarized_results

    def inferrer_results(self, inferrer, observation, leaf_scores):
        # aggregation for calculating a common ancestor, using only vision scores
        vision_aggregated_scores = inferrer.aggregate_results(
            leaf_scores,
            score_ratio_cutoff=InatInferrer.COMMON_ANCESTOR_CUTOFF_RATIO,
            max_leaf_scores_to_consider=InatInferrer.COMMON_ANCESTOR_WINDOW,
            column_for_cutoff="vision_score"
        )
        # aggregation for calculating a common ancestor, using combined scores
        combined_aggregated_scores = inferrer.aggregate_results(
            leaf_scores,
            score_ratio_cutoff=InatInferrer.COMMON_ANCESTOR_CUTOFF_RATIO,
            max_leaf_scores_to_consider=InatInferrer.COMMON_ANCESTOR_WINDOW,
            column_for_cutoff="combined_score"
        )

        # calculate common ancestors and scores for both vision-only, and combined scores
        vision_common_ancestor = inferrer.common_ancestor_from_aggregated_scores(
            vision_aggregated_scores, score_to_use="vision_score"
        )

        combined_common_ancestor = inferrer.common_ancestor_from_aggregated_scores(
            combined_aggregated_scores, score_to_use="combined_score"
        )

        # record the top 100 scores of 3 score types: vision only, combined, and combined + nearby
        vision_top_100 = leaf_scores.sort_values(
            "vision_score", ascending=False
        ).reset_index(drop=True).head(100)

        combined_top_100 = leaf_scores.sort_values(
            "combined_score", ascending=False
        ).reset_index(drop=True).head(100)

        combined_nearby_top_100 = leaf_scores.query(
            "geo_score >= geo_threshold"
        ).sort_values(
            "combined_score", ascending=False
        ).reset_index(drop=True).head(100)

        return {
            "vision": {
                "scores": vision_top_100,
                "common_ancestor": {
                    "taxon": vision_common_ancestor,
                    "pool_size": self.common_ancestor_pool_size(vision_aggregated_scores),
                    "score": self.common_ancestor_score(vision_common_ancestor)
                }
            },
            "combined": {
                "scores": combined_top_100,
                "common_ancestor": {
                    "taxon": combined_common_ancestor,
                    "pool_size": self.common_ancestor_pool_size(combined_aggregated_scores),
                    "score": self.common_ancestor_score(combined_common_ancestor)
                }
            },
            "combined_nearby": {
                "scores": combined_nearby_top_100,
                "common_ancestor": {
                    "taxon": combined_common_ancestor,
                    "pool_size": self.common_ancestor_pool_size(combined_aggregated_scores),
                    "score": self.common_ancestor_score(combined_common_ancestor)
                }
            }
        }

    def common_ancestor_score(self, common_ancestor):
        return common_ancestor[
            "normalized_aggregated_combined_score"
        ] if common_ancestor is not None else 0

    def common_ancestor_pool_size(self, aggregated_scores):
        return aggregated_scores.query(
            "leaf_class_id.notnull()"
        ).index.size

    def summarize_result_subset(
        self, inferrer, observation, inferrer_results, summary_index, cutoff=False
    ):
        working_results = inferrer_results["scores"]
        score_column = "vision_score" if summary_index == "vision" else "combined_score"
        values = working_results.head(1)[score_column].values
        if len(values) == 0:
            top_score = 0
        else:
            top_score = values[0]
        if cutoff:
            working_results = working_results.query(
                f"{score_column} > {top_score * 0.001}"
            ).head(10)

        normalized_score_column = f"normalized_{score_column}"
        normalized_values = working_results.head(1)[normalized_score_column].values
        if len(values) == 0:
            top_normalized_score = 0
        else:
            top_normalized_score = normalized_values[0]

        summary = {}
        common_ancestor = inferrer_results["common_ancestor"]
        if common_ancestor["taxon"] is None:
            summary["common_ancestor_id"] = 0
            summary["common_ancestor_score"] = 0
            summary["common_ancestor_pool_size"] = 0
            summary["common_ancestor_ancestors"] = ""
            summary["common_ancestor_rank_level"] = np.nan
            summary["common_ancestor_accurate"] = 0
            summary["common_ancestor_present"] = 0
        else:
            is_accurate = common_ancestor["taxon"].taxon_id in observation.taxon_ancestry
            summary["common_ancestor_id"] = common_ancestor["taxon"].taxon_id
            summary["common_ancestor_score"] = common_ancestor["score"]
            summary["common_ancestor_pool_size"] = common_ancestor["pool_size"]
            summary["common_ancestor_ancestors"] = "/".join(
                str(a) for a in
                inferrer.taxonomy.taxon_ancestors[
                    common_ancestor["taxon"].taxon_id
                ]
            )
            summary["common_ancestor_rank_level"] = common_ancestor["taxon"].rank_level
            summary["common_ancestor_accurate"] = 1 if is_accurate else 0
            summary["common_ancestor_present"] = 1

        matching_index = self.matching_index(observation, working_results)

        results_count = len(working_results.index)
        summary["results_count"] = results_count
        summary["matching_index"] = matching_index
        summary["recall"] = 1 if matching_index is not None else 0
        summary["precision"] = 0 if matching_index is None else 1 / results_count

        sum_of_precision_and_recall = summary["precision"] + summary["recall"]
        summary["f1"] = 0 if sum_of_precision_and_recall == 0 else (
            2 * summary["precision"]
        ) / sum_of_precision_and_recall

        summary["top_score"] = top_normalized_score
        summary["matching_score"] = self.matching_score(observation, working_results, normalized_score_column)

        return summary

    def matching_index(self, observation, results):
        matching_indices = results.index[
            results["taxon_id"] == observation.taxon_id
        ].tolist()
        return matching_indices[0] if len(matching_indices) > 0 else None

    def matching_score(self, observation, results, score_column):
        matches = results.query(
            f"taxon_id == {observation.taxon_id}"
        )
        values = matches.head(1)[score_column].values
        if len(values) == 0:
            return 0
        else:
            return values[0]

    async def download_photo_async(self, photo_url):
        checksum = hashlib.md5(photo_url.encode()).hexdigest()
        cache_path = os.path.join(self.upload_folder, "obs-" + checksum) + ".jpg"
        if await aiofiles.os.path.exists(cache_path):
            return cache_path
        async with self.session.get(photo_url) as resp:
            if resp.status == 200:
                f = await aiofiles.open(cache_path, mode="wb")
                await f.write(await resp.read())
                await f.close()
        if not os.path.exists(cache_path):
            return
        mime_type = magic.from_file(cache_path, mime=True)
        if mime_type != "image/jpeg":
            im = Image.open(cache_path)
            rgb_im = im.convert("RGB")
            rgb_im.save(cache_path)
        return cache_path

    def debug(self, message):
        if self.cmd_args["debug"]:
            print(message)

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
