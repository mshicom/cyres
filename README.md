cyres
=====
Python bindings for ceres-solver (via Cython)
##How to use
Suppose you have already installed the ceres in /usr/local.
Now first build &  install the main library:
```
cd cyres/
python setup.py install
```
After this you should be able to import the cyres module in python, like:
```
python
import cyres
```
which should give you no errors. Then you can build the example code for the cost function:
```
cd examples/quadratic/
python setup.py build_ext -i
```
Now you can try out the example:
```
python quadratic.py
```