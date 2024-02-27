import click
import json
import asyncio


@click.command()
@click.option("--exclude-train-photos-path", "-x", multiple=True,
              help="Exclude photos that were included in these training sets.")
@click.option("--limit", type=int, show_default=True, default=5000,
              help="Number of observations to include.")
@click.option("--standard_set", help="Export the standard set of 18 filtered datasets.")
@click.option("--filename_suffix", type=str, help="String to add to end of filename.")
@click.option("--place_id", type=int, help="Export observations in this place.")
@click.option("--taxon_id", type=int, help="Export observations in this taxon.")
def test(**args):
    # some libraries are slow to import, so wait until command is validated and properly invoked
    from lib.model_test_data_export_manager import ModelTestDataExportManager
    print("\nArguments:")
    print(json.dumps(args, indent=4))
    print("\nInitializing ModelTestDataExporter...\n")
    model_test_data_exporter = ModelTestDataExportManager(**args)
    print("Exporting data...\n")
    if "standard_set" in args and args["standard_set"]:
        asyncio.run(model_test_data_exporter.generate_standard_set())
    else:
        asyncio.run(model_test_data_exporter.generate_from_cmd_args())
    print("\nDone\n")


if __name__ == "__main__":
    test()
