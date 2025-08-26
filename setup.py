from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.1.0"

REPO_NAME = "WOULD_WE_HAVE_MET"
AUTHOR_USER_NAME = "yeswanth2715"
PACKAGE_NAME = "wouldtheyhavemet"  # must match src/wouldtheyhavemet
AUTHOR_EMAIL = "jyeswanthreddy@gmail.com"

setup(
    name=PACKAGE_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Serendipity modeling: Would We Have Met?",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
