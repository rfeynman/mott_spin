# retarding Mott data analysis
This code take the retarding field Mott polarimeter data(cvs file). Two evaluation method are included:
* Do the standard procedure, fit the sherman function up to the 0 energy loss. Then evalaution polarition. 
* Fast polarization evaluation by using Sherman functionat single energy loss point
### mott_v1.py

* analyze the spin polarization data. find the effective sherman function.  split the historical data and test data.
  *evalaute the polarization using effect sherman function

### mott_v1.1.py

* add error function, polarization calculation has the error bar

### mott.py

* fix the error propgate,
* add energy loss scan,
* break down the main() by creating two functions
