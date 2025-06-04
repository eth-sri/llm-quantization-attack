from setuptools import setup, find_packages

setup(
    name="q_attack",
    version="0.0.1",
    packages=find_packages(include=["q_attack", "q_attack.*"]),
)
