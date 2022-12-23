import importlib


class InvalidFactoryArgument(Exception):
    def __init__(self, argument: str, msg: str):
        self.argument = argument
        self.message = msg
        super().__init__(self.message)


def create_instance_from_string(module_class_name: str):
    # Guard clause
    if module_class_name is None:
        raise InvalidFactoryArgument(
            argument=module_class_name,
            msg=f"model class not provided to the factory. argument: {module_class_name}",
        )

    # Get the module
    module_name = None
    class_name = None
    try:
        module_name, class_name = module_class_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
    except (AttributeError, ModuleNotFoundError):
        raise InvalidFactoryArgument(
            argument=module_name,
            msg=f"Invalid module provided to the factory. argument: {module_name}",
        )

    # Get the class
    try:
        created_class = getattr(module, class_name)
    except AttributeError:
        raise InvalidFactoryArgument(
            argument=class_name,
            msg=f"Invalid class provided to the factory. argument: {class_name}",
        )

    # Return an instance of the class
    return created_class()
