import pytest
from image_classification import retrieve_configs, InvalidConfigFileTypeError

def test_n_layer_args_happy_case():

    config_file = 'tests/resources/n_layer_happy.yaml'
    vals = retrieve_configs(config_file)
    assert 'n_layer_nn' in vals.get('model_class')
    assert [12288, 20, 7, 5, 1] == vals.get('config')['layers_dims']

def test_retrieve_configs_args_no_config():
    config_file = None
    with pytest.raises(InvalidConfigFileTypeError):
        retrieve_configs(config_file)

def test_invalid_config_filetype():
    config_file = 'tests/resources/not_a_yaml'
    vars = retrieve_configs(config_file)
    assert vars is None

def test_retrieve_configs_invalid_yaml():
    config_file = 'tests/resources/invalid.yaml'
    with pytest.raises(InvalidConfigFileTypeError):
        retrieve_configs(config_file)