import yaml
import json
import re
import os
import click
import pandas as pd

def reconcile(gemini_file_path, vision_file_path, result_file_path):
    print("Reconcile:")
    print(gemini_file_path)
    print(vision_file_path)
    print(result_file_path)

    gemini_file = pd.read_csv(gemini_file_path)
    vision_file = pd.read_csv(vision_file_path)

    # Filter vision_file where method is "combined"
    vision_file_filtered = vision_file[(vision_file['method'] == 'combined') & (vision_file['inferrer_name'] == 2.15)]
    
    # Merge gemini_file with vision_file_filtered on observation_id == uuid
    merged_df = pd.merge(gemini_file, vision_file_filtered[['uuid', 'matching_index']], 
                        left_on='observation_id', right_on='uuid', 
                        how='left')

    # Drop the redundant 'uuid' column if necessary
    merged_df.drop('uuid', axis=1, inplace=True)

    # Add a new column 'is_matching_index_zero' which is True if matching_index == 0, otherwise False
    merged_df['is_matching_index_zero'] = merged_df['matching_index'] == 0

    # Add another column 'is_matching_index_zero_int' which is 1 if True, 0 if False
    merged_df['is_matching_index_zero_int'] = merged_df['is_matching_index_zero'].astype(int)

    # Save the result to a new CSV file
    merged_df.to_csv(result_file_path, index=False)

@click.command()
@click.option("--data_dir", type=click.Path(), help="Path to test data CSVs directory.")
@click.option("--label", required=True, type=str, help="Label used for output.")
def test(**args):
    print("\nArguments:")
    print(json.dumps(args, indent=4))
    
    label = args["label"]
    folder_path = args["data_dir"]

    for file in sorted(os.listdir(folder_path)):
        eval_filename_match = re.search(rf"{label}-test-results-[0-9]{{8}}-([a-zA-Z]+)-.*\.csv", file)
        if eval_filename_match:
            group = eval_filename_match.group(1)
            for observation_file in sorted(os.listdir(folder_path)):
                if re.search(f"test-results-[0-9]{{8}}-{group}-.*-observations\.csv", observation_file):
                    reconcile(
                        os.path.join(folder_path, file), 
                        os.path.join(folder_path, observation_file),
                        os.path.join(folder_path, "comparison-"+file)
                    )


if __name__ == "__main__":
    test()
