pip install wheel
python setup.py bdist_wheel
pip install twine
twine upload dist/*
rm -rf build
rm -rf dist
rm -rf sixdrepnet.egg-info