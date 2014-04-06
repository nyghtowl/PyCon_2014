# PyCon 2014 Example Linear Regression

This is a linear regression example shown in my 'How to get started with Machine Learning' PyCon 2014 talk. 


Files
--------

Main script written in lr_brain_body.py. The ipython notebook provides example code snippets and show the development of this example.

All png files are illustration outputs used in the presentation.


Dataset
--------

Dataset:  brainhead.dat
http://www.stat.ufl.edu/~winner/data/brainhead.dat

Source: R.J. Gladstone (1905). "A Study of the Relations of the Brain to 
to the Size of the Head", Biometrika, Vol. 4, pp105-123

Description: Brain weight (grams) and head size (cubic cm) for 237
adults classified by gender and age group.

Variables/Columns
Gender   8   /* 1=Male, 2=Female  */
Age Range  16   /* 1=20-46, 2=46+  */
Head size (cm^3)  21-24 - converted to inches cubed
Brain weight (grams)  29-32 - converted to pounds



First Draft
--------

Built initial linear regression with code taken straight from sci-kit learn.

- Code source: Jaques Grobler
- License: BSD 3 clause
- Code has been modified since
