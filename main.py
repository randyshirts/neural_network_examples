import argparse
import importlib

import utils.configuration as configuration


def main():
    # Initialize parser
    msg = "Executes image classification to identify cat vs non-cat images."
    parser = argparse.ArgumentParser(description=msg)

    # Adding optional argument
    parser.add_argument("-c", "--Config", help="Configuration file path")

    # Read arguments from command line
    args = parser.parse_args()

    # Parse yaml to obtain config values as dict
    config_values = configuration.retrieve_configs(args.Config) if args.Config else None

    # Train the model, evaluate, and sample the trained model
    trainer = importlib.import_module(config_values["trainer"])
    trainer.train(config_values)


if __name__ == "__main__":
    main()
