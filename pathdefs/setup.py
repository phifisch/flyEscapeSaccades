import setuptools

setuptools.setup(
    name="pathdefs_biofish", # Replace with your own username
    version="0.1",
    author="You",
    description="path definitions for work",
    long_description="Minimal package with only few path definitions, i.e. path to data and source code directories. The paths can be defined on each machine independently without changing the notebook code files anymore",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)