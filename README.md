

Recommended visualization format: *.ipynb
Additional formats provided in *.py and *.PDF


This exercise has been divided in 3 files:

IRS990_Tech_Exercise_DATA_ACCESS (.pynb/.py) contains parts 1. Data Access and 2. Feature Extraction. In this file, I access AWS’s  database for Form 990 filings, download select features as outlined in the python notebook annotation, and save the output as a csv file.

IRS990_Tech_Exercise_CLUSTERING (.pynb/.py) contains parts 3. Modeling and 4. Visualization, 
using a K-means algorithm to cluster the database, filtered for 501(c)(3) entities only, based on
revenue, expenses and compensation data. Result visualization and analysis are also included.

Input  folder contains csv files obtained from AWS (index of filings, and data frames build from XML data of the filings).
Output folder contains PDF files generated from the codes above.
