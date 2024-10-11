from setuptools import setup, find_packages

setup(
    name="progs",  # Name of your package
    version="0.1",  # Version of your package
    packages=find_packages(),  # Automatically find all packages in the 'progs' directory
    include_package_data=True,  # Include non-code files from MANIFEST.in (if any)
    
    # Metadata about the package
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for image processing including reading, displaying, and processing images",

    # Dependencies, if any
    install_requires=[
        # Add external libraries your project depends on
        "numpy>=1.18.0",  # Example dependency, modify according to your needs
        "Pillow>=8.0.0"
    ],

    # Python version compatibility
    python_requires=">=3.6",  # Specify the minimum Python version supported

    # Optional: Entry points to create command-line tools
)

