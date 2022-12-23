import pytest

from utils.factory import InvalidFactoryArgument, create_instance_from_string


def test_model_factory_happy_case():
    model_class = (
        "feed_forward.image_classification.manual.n_layer.n_layer_nn.N_Layer_NN"
    )
    model = create_instance_from_string(model_class)
    assert model


def test_model_factory_no_class_provided():
    model_class = None
    with pytest.raises(InvalidFactoryArgument):
        create_instance_from_string(model_class)


def test_model_factoy_invalid_module():
    model_class = "feed_forward.image_classification.N_Layer_NN"
    with pytest.raises(InvalidFactoryArgument):
        create_instance_from_string(model_class)


def test_model_factoy_invalid_class():
    model_class = (
        "feed_forward.image_classification.manual.n_layer.n_layer_nn.invalid_class"
    )
    with pytest.raises(InvalidFactoryArgument):
        create_instance_from_string(model_class)
