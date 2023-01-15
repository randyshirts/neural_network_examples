import matplotlib.pyplot as plt
import numpy as np

import utils.factory as factory
from feed_forward.image_classification.data_loader.data_loader import (
    flatten_image_data,
    load_data,
)


def train(configs: dict = None) -> None:

    # Set default values
    if configs is None:
        configs = {}
        configs["model_class"] = "manual.n_layer.n_layer_nn.N_Layer_NN"
        configs["config"] = {}
        configs["config"]["layers_dims"] = [12288, 20, 7, 5, 1]
        configs["config"]["learning_rate"] = 0.0075
        configs["config"]["num_iterations"] = 3000
        configs["config"]["print_cost"] = False

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

    # Get model parameters from config file
    model_config = configs["config"]
    if "learning_rate" in model_config:
        lr = model_config["learning_rate"]
    if "num_iterations" in model_config:
        num_iterations = model_config["num_iterations"]
    print_cost = model_config.get("print_cost", False)
    if "n_x" in model_config:
        n_x = model_config["n_x"]
    if "n_h" in model_config:
        n_h = model_config["n_h"]
    if "n_y" in model_config:
        n_y = model_config["n_y"]
    if "layers_dims" in model_config:
        layers_dims = model_config["layers_dims"]
    if "Two_Layer_NN" in configs["model_class"]:
        layers_dims = (n_x, n_h, n_y)

    # Create a model object based on the config values
    model = factory.create_instance_from_string(configs["model_class"])

    # Train the model
    parameters = model.model(
        train_x,
        train_y,
        layers_dims,
        learning_rate=lr,
        num_iterations=num_iterations,
        print_cost=print_cost,
    )
    # Evaluate the model based on train data and then test data
    _ = model.predict(train_x, train_y, parameters)
    _ = model.predict(test_x, test_y, parameters)

    # Test an image external to train and test data
    num_px = train_x_orig.shape[1]
    model.eval.sample_external_image(
        num_pixels=num_px, parameters=parameters, classifications=classes, model=model
    )
