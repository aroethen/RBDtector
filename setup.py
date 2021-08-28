import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RBDtector",
    version="0.0.1",
    author="Annika Röthenbacher",
    description="A package to detect motoric arousal consistent with REM sleep behaviour disorder.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "RBDtector"},
    packages=setuptools.find_packages(where="RBDtector"),
    python_requires=">=3.6"
)
