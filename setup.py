from setuptools import setup, find_packages

setup(
    name="nemo-speechlm-vllm",
    version="0.2.0",
    packages=find_packages(),
    install_requires=["nemo_toolkit[asr]"],
    entry_points={
        "vllm.general_plugins": [
            "nemo_speechlm = nemo_speechlm:register",
        ],
    },
)
