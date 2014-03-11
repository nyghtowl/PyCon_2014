Example linear regression model for PyCon 2014. In progress of building out

Initial code taken straight from sci-kit learn
- Code source: Jaques Grobler
- License: BSD 3 clause
- Code has been modified since

***************

Datasets

Captured dataset for brain & body weight from: http://people.sc.fsu.edu/~jburkardt/datasets/regression/x01.txt

The data records the average weight of the brain and body for a number of mammal species.  

There are 62 rows of data.  The 3 data columns include:
- I,  the index,
- A1, the brain weight;
- B,  the body weight.


Dataset comparing population to drinking
http://people.sc.fsu.edu/~jburkardt/datasets/regression/x20.txt

Discussion:

In various states, population and drinking data was recorded. There are 46 rows of data.  The data includes:
- I,  the index;
- A0, 1;
- A1, the size of the urban population,
- A2, the number of births to women between 45 to 49
-     (actually, the reciprocal of that value, times 100)
- A3, the consumption of wine per capita,
- A4, the consumption of hard liquor per capita,
- B,  the death rate from cirrhosis.



Dataset:  brainhead.dat
http://www.stat.ufl.edu/~winner/data/brainhead.dat

Source: R.J. Gladstone (1905). "A Study of the Relations of the Brain to 
to the Size of the Head", Biometrika, Vol. 4, pp105-123

Description: Brain weight (grams) and head size (cubic cm) for 237
adults classified by gender and age group.

Variables/Columns
Gender   8   /* 1=Male, 2=Female  */
Age Range  16   /* 1=20-46, 2=46+  */
Head size (cm^3)  21-24
Brain weight (grams)  29-32