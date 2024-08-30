import click
import yaml
import json
import asyncio

@click.command()
@click.option("--data_dir", type=click.Path(), help="Path to test data CSVs directory.")
@click.option("--label", required=True, type=str, help="Label used for output.")
@click.option("--limit", type=int, show_default=True, default=100, help="Max number of observations to test.")
def test(**args):
    print("\nArguments:")
    print(json.dumps(args, indent=4))
    
    from evaluate_gemini_results import GeminiEvalutation
    geminiEvalutation = GeminiEvalutation(**args)

    asyncio.run(geminiEvalutation.run_async())

    print("\nDone\n")


if __name__ == "__main__":
    test()
