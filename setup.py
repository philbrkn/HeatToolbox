from setuptools import setup, find_packages

setup(
    name="heatoptim",
    version="0.1.0",
    description="toolbox for heat-related simulations and processing.",
    author="Placeholder",
    author_email="email@example.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",    
        "matplotlib==3.9.2",
        "mpi4py",
        # "dolfinx",
        "gmsh==4.13.1",
        "cma==3.2.2",
        "pymoo==0.6.1.3",
        "torch==2.2.2",
        "torchaudio==2.2.2",
        "torchvision==0.17.2",
        "vtk==9.3.1"
    ],
    package_data={
        # If you need to include non-code assets (like VAE models) inside the package:
        "heatoptim": ["models/*"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "heatoptim=heatoptim.run_gui:main",
        ],
    },
)
