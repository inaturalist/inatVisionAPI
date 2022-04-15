import click
import yaml

CONFIG = yaml.safe_load(open("config.yml"))


@click.command()
@click.option("--path", required=True, type=click.Path(), help="Path to test data CSV.")
@click.option("--limit", type=int, show_default=True, default=100,
              help="Max number of observations to test.")
@click.option("--observation_id", type=int, help="Single observation ID to test.")
@click.option("--filter-iconic/--no-filter-iconic", show_default=True, default=True,
              help="Use iconic taxon for filtering.")
@click.option("--print-tree", is_flag=True, show_default=True, default=False,
              help="Print trees for results.")
@click.option("--debug", is_flag=True, show_default=True, default=False,
              help="Output debug messages.")
def test(**args):
    # some libraries are slow to import, so wait until command is validated and properly invoked
    from lib.vision_testing import VisionTesting
    print("\nArguments:")
    print(args)
    print("\nInitializing VisionTesting...\n")
    VisionTesting(CONFIG, **args)
    print("\nDone\n")


if __name__ == "__main__":
    test()

