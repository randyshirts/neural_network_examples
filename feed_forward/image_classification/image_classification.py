import argparse
import numpy as np
import importlib
import matplotlib.pyplot as plt
import yaml
from dnn_utils import load_data


class InvalidConfigFileTypeError(Exception):
    def __init__(self, config_file: str, msg: str) -> None:
        self.config_file = config_file
        self.message = msg
        super().__init__(self.message)


def show_example_image(
    index: int = 15, training_data=None, training_labels=None, classifications=None
) -> None:
    # Show an example picture
    # Example of a picture
    plt.imshow(training_data[index])
    plt.show()

    print(
        "y = "
        + str(training_labels[0, index])
        + ". It's a "
        + classifications[training_labels[0, index]].decode("utf-8")
        + " picture."
    )


def display_dataset_info(training_data=None, training_labels=None, test_labels=None):
    # Explore your dataset
    m_train = training_data.shape[0]
    num_px = training_data.shape[1]
    m_test = training_data.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(training_data.shape))
    print("train_y shape: " + str(training_labels.shape))
    print("test_x_orig shape: " + str(training_data.shape))
    print("test_y shape: " + str(test_labels.shape))


def flatten_image_data(original_images: list[int, int, int] = None) -> list:
    # Reshape the training and test examples
    flattened_data = original_images.reshape(
        original_images.shape[0], -1
    ).T  # The "-1" makes reshape flatten the remaining dimensions

    # Standardize data to have feature values between 0 and 1.
    flattened_image_features = flattened_data / 255.0

    return flattened_image_features


def sample_external_image(
    num_pixels: int, parameters: list, classifications: list, model: object
) -> None:
    import imageio.v2 as imageio
    import PIL

    my_image = "IMG_2296.jpg"  # change this to the name of your image file
    my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)

    fname = "images/" + my_image
    image = np.array(imageio.imread(fname))
    my_image = np.array(
        PIL.Image.fromarray(image).resize((num_pixels, num_pixels))
    ).reshape((num_pixels * num_pixels * 3, 1))
    my_predicted_image = model.predict(my_image, my_label_y, parameters)

    plt.imshow(image)
    print(
        "y = "
        + str(np.squeeze(my_predicted_image))
        + ', your L-layer model predicts a "'
        + classifications[
            int(np.squeeze(my_predicted_image)),
        ].decode("utf-8")
        + '" picture.'
    )
    plt.show()


def train(configs: dict = None) -> None:

    # Set default values
    if configs is None:
        configs = {}
        configs["model_class"] = "n_layer_nn.N_Layer_NN"
        configs["config"] = {}
        configs["config"]["layers_dims"] = [12288, 20, 7, 5, 1]

    # matplotlib inline
    plt.rcParams["figure.figsize"] = (5.0, 4.0)  # set default size of plots
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "gray"

    np.random.seed(55555)

    # Load the data
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # show_example_image(index=25, training_data=train_x_orig, training_labels=train_y, classifications=classes)

    # display_dataset_info(training_data=train_x_orig, training_labels=train_y, test_labels=test_y)

    train_x = flatten_image_data(original_images=train_x_orig)
    test_x = flatten_image_data(original_images=test_x_orig)
    # print ("train_x's shape: " + str(train_x.shape))
    # print ("test_x's shape: " + str(test_x.shape))

    ### Get model parameters from config file ####
    if "n_x" in configs["config"]:
        n_x = configs["config"]["n_x"]
    if "n_h" in configs["config"]:
        n_h = configs["config"]["n_h"]
    if "n_y" in configs["config"]:
        n_y = configs["config"]["n_y"]
    if "layers_dims" in configs["config"]:
        layers_dims = configs["config"]["layers_dims"]
    if "Two_Layer_NN" in configs["model_class"]:
        layers_dims = (n_x, n_h, n_y)

    # Create a model object based on the config values
    model = model_factory(configs["model_class"])

    # Train the model
    parameters = model.model(
        train_x, train_y, layers_dims, num_iterations=2500, print_cost=True
    )
    # Evaluate the model based on train data and then test data
    pred_train = model.predict(train_x, train_y, parameters)
    pred_test = model.predict(test_x, test_y, parameters)

    # Test an image external to train and test data
    num_px = train_x_orig.shape[1]
    sample_external_image(
        num_pixels=num_px, parameters=parameters, classifications=classes, model=model
    )


class InvalidModelFactoryArgument(Exception):
    def __init__(self, argument: str, msg: str):
        self.argument = argument
        self.message = msg
        super().__init__(self.message)


def model_factory(model_class_name: str) -> object:
    """
    Creates a model instance from a string
    """
    # Guard clause
    if model_class_name is None:
        raise InvalidModelFactoryArgument(
            argument=model_class_name,
            msg=f"model class not provided to the model factory. argument: {model_class_name}",
        )

    # Get the module
    module_class = None
    class_name = None
    try:
        module_name, class_name = model_class_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
    except (AttributeError, ModuleNotFoundError):
        raise InvalidModelFactoryArgument(
            argument=model_class_name,
            msg=f"Invalid module provided to the model factory. argument: {model_class_name}",
        )

    # Get the class
    try:
        model_class = getattr(module, class_name)
    except AttributeError:
        raise InvalidModelFactoryArgument(
            argument=class_name,
            msg=f"Invalid class provided to the model factory. argument: {class_name}",
        )

    # Return an instance of the class
    return model_class()


def retrieve_configs(config_file: str) -> dict:
    """
    Takes a file path and returns a dictionary
    """
    config_values = None
    try:
        with open(config_file, "r") as f:
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


def main():
    # Initialize parser
    msg = "Executes image classification to identify cat vs non-cat images."
    parser = argparse.ArgumentParser(description=msg)

    # Adding optional argument
    parser.add_argument("-c", "--Config", help="Configuration file path")

    # Read arguments from command line
    args = parser.parse_args()

    # Parse yaml to obtain config values as dict
    config_values = retrieve_configs(args.Config) if args.Config else None

    # Train the model, evaluate, and sample the trained model
    train(config_values)


if __name__ == "__main__":
    main()
