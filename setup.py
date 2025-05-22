from setuptools import setup, find_packages

setup(
    name="repairs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "brax>=0.9.0",
        "numpy>=1.22.0",
        "ipython",
    ],
    python_requires=">=3.8",
)
