Simple logistic regression on a dataset obtained from [https://global.health](https://global.health).


Dependencies are Python 3, `numpy`, `pandas`, `scikit-learn` and `matplotlib`. Install them by running `pip3 install -r requirements.txt`.


Instructions:
1. register on [https://global.health](https://global.health) and download the COVID-19 dataset (~500 Mb)
2. create a folder `data` in the root directory of this repository
3. copy the downloaded `.tar` file in there
4. extract the tar file (on the Linux command ine, e.g., via `tar -xf <filename>.tar`
5. run the data extraction / preprocessing script via `python extract_data.py`
6. run the logistic regression script via `python __init__.py`
