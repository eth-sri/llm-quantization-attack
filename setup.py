from setuptools import find_packages, setup

setup(
    name="q_attack",
    version="0.0.1",
    author="Kazuki Egashira",
    author_email="kegashira@ethz.ch",
    packages=find_packages(include=['q_attack', 'q_attack.*']),
    install_requires=[]
)
