import os
import re
import itertools
import tempfile
import termcolor
import string
import random
import codecs
from pathlib import Path
import yaml


def _safe_str(name, length=8):
    name = re.sub(r"(\W)\1+", r"\1", re.sub(r"[^a-zA-Z0-9]", "-", name))
    if len(name) > length:
        name = name[:length]
    else:
        name += "a" * (length - len(name))
    return name.lower()


def _encode_name(name):
    uid = "".join(random.choice(string.ascii_lowercase) for _ in range(4))
    return codecs.encode((f"{_safe_str(name)}-{uid}").replace("-", ""), "rot13")


def _read_text(fp):
    with open(fp, "r") as f:
        text = f.read()
    return text


def _submit(args, name):
    # context = _get_current_context()
    template = _read_text(f"{os.path.dirname(__file__)}/job_template.yaml")
    wandb_key = _read_text(f"{os.path.dirname(__file__)}/wandb.key")
    cfg = yaml.safe_load(
        Path(f"{os.path.dirname(__file__)}/container_config.yaml").read_text()
    )
    cmd = cfg.get("cmd", None)
    assert cmd is not None, f"No cmd found in config:\n{cfg}"
    cfg.update(
        name="alopez-" + _encode_name(name),
        # namespace=context,
        wandb_key=wandb_key,
        cmd=" ".join([cmd, args]),
    )
    while "{{" in template:
        for key, value in cfg.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            template = re.sub(regexp, str(value), template)
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml")
    with open(tmp.name, "w") as f:
        f.write(template)
    print(termcolor.colored(f'{cfg["name"]}', "yellow"), cfg["cmd"])
    os.system(f"kubectl create -f {tmp.name}")
    tmp.close()


def _submit_batch(kwargs: dict):
    tasks = kwargs.pop("tasks", None)
    if tasks:
        tasks = "[" + ",".join(tasks) + "]"
    arg_list = list(itertools.product(*kwargs.values()))
    if len(arg_list) > 16:
        print(
            termcolor.colored(f"Error: {len(arg_list)} jobs exceeds limit of 6", "red")
        )
        return
    print(
        termcolor.colored(
            f'Submitting {len(arg_list)} job{"s" if len(arg_list) > 1 else ""}', "green"
        )
    )
    for args in arg_list:
        args = " ".join([f"{k}={v}" for k, v in zip(kwargs.keys(), args)])
        if tasks:
            args += f" tasks={tasks}"
        _submit(
            args, name=kwargs["ex p_name"][0] if "exp_name" in kwargs else "default"
        )


if __name__ == "__main__":
    # Load the arguments from the batch_job_config.yaml file
    kwargs = yaml.safe_load(
        Path(f"{os.path.dirname(__file__)}/batch_job_config.yaml").read_text()
    )
    _submit_batch(kwargs)
