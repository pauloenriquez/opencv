from setuptools import setup, find_packages

setup(
    name='opencv',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'Pillow'
        'streamlit'
        # other dependencies
    ],
)
