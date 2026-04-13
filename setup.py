from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    requirement_lst: List[str] = []
    try:
        with open('requirements.txt') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print('requirements.txt file not found')

    return requirement_lst

setup(
    name = 'Air_Pollution',
    version = '0.0.1',
    author = 'ArgusX',
    author_email = 'contact@argusx.com.mx',
    packages = find_packages(),
    install_requires = get_requirements()
)