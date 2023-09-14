import json
import subprocess
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, call, ANY

import pytest
import torch


@pytest.mark.parametrize("version", ("v1", "v2"))
def test_main(fake_checkpoint_dir, monkeypatch, version):
    if version == "v1":
        import generate.adapter as generate
    else:
        import generate.adapter_v2 as generate

    config_path = fake_checkpoint_dir / "lit_config.json"
    config = {"block_size": 16, "vocab_size": 50, "n_layer": 2, "n_head": 4, "n_embd": 8, "rotary_percentage": 1}
    config_path.write_text(json.dumps(config))

    load_mock = Mock()
    load_mock.return_value = load_mock
    load_mock.__enter__ = Mock()
    load_mock.__exit__ = Mock()
    monkeypatch.setattr(generate, "lazy_load", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.encode.return_value = torch.tensor([[1, 2, 3]])
    tokenizer_mock.return_value.decode.return_value = "### Response:foo bar baz"
    monkeypatch.setattr(generate, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor([[3, 2, 1]])
    monkeypatch.setattr(generate, "generate", generate_mock)

    num_samples = 1
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        generate.main(temperature=2.0, top_k=2, checkpoint_dir=fake_checkpoint_dir)

    assert len(tokenizer_mock.return_value.decode.mock_calls) == num_samples
    assert torch.allclose(tokenizer_mock.return_value.decode.call_args[0][0], generate_mock.return_value)
    assert (
        generate_mock.mock_calls
        == [call(ANY, ANY, ANY, max_seq_length=101, temperature=2.0, top_k=2, eos_id=ANY)] * num_samples
    )
    # only the generated result is printed to stdout
    assert out.getvalue() == "foo bar baz\n" * num_samples

    assert "'padded_vocab_size': 512, 'n_layer': 2, 'n_head': 4, 'n_embd': 8" in err.getvalue()


@pytest.mark.parametrize("version", ("", "_v2"))
def test_cli(version):
    cli_path = Path(__file__).parent.parent / "generate" / f"adapter{version}.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Generates a response" in output
