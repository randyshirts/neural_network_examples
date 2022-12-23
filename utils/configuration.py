import yaml


class InvalidConfigFileTypeError(Exception):
    def __init__(self, config_file: str, msg: str) -> None:
        self.config_file = config_file
        self.message = msg
        super().__init__(self.message)


def retrieve_configs(config_file: str) -> dict:
    """
    Takes a file path and returns a dictionary
    """
    config_values = None
    try:
        with open(config_file) as f:
            config_values = yaml.load(f, Loader=yaml.FullLoader)
    except TypeError:
        raise InvalidConfigFileTypeError(
            config_file=config_file,
            msg=f"Expected a config file of type yaml. file: {config_file}",
        )
    except yaml.scanner.ScannerError:
        raise InvalidConfigFileTypeError(
            config_file=config_file,
            msg=f"Invalid yaml within supplied config file. file: {config_file}",
        )
    return config_values
