import click
import yaml
import json
import asyncio

CONFIG = yaml.safe_load(open("config.yml"))


@click.command()
@click.option("--path", type=click.Path(), help="Path to test data CSV.")
@click.option("--data_dir", type=click.Path(), help="Path to test data CSVs directory.")
@click.option("--label", required=True, type=str, help="Label used for output.")
@click.option("--limit", type=int, show_default=True, default=100,
              help="Max number of observations to test.")
@click.option("--geo/--no-geo", show_default=True, default=True,
              help="Use geo model.")
@click.option("--observation_id", type=str, help="Single observation UUID to test.")
@click.option("--filter-iconic/--no-filter-iconic", show_default=True, default=True,
              help="Use iconic taxon for filtering.")
@click.option("--debug", is_flag=True, show_default=True, default=False,
              help="Output debug messages.")
def test(**args):
    if not args["path"] and not args["data_dir"]:
        print("\nYou must specify either a `--path` or a `--data_dir` option\n")
        exit()

    # some libraries are slow to import, so wait until command is validated and properly invoked
    from lib.vision_testing import VisionTesting
    print("\nArguments:")
    print(json.dumps(args, indent=4))
    print("\nInitializing VisionTesting...\n")
    testing = VisionTesting(CONFIG, **args)

    asyncio.run(testing.run_async())

    print("\nDone\n")


if __name__ == "__main__":
    test()
