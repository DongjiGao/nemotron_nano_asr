from setuptools import setup, find_packages

setup(
    name="nemotron-nano-asr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["nemo_toolkit[asr]"],
    entry_points={
        "vllm.general_plugins": [
            "nemotron_nano_asr = nemotron_nano_asr:register",
        ],
    },
)
