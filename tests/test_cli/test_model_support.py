from transformers_cfg.cli.cli_main import check_model_support


def test_supported_model():
    model = "gpt2"
    assert check_model_support(model) == True


def test_unsupported_model():
    model = "bigscience/bloom"
    assert check_model_support(model) == False
