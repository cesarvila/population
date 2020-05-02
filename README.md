# The information behind COVID-19 Data

### Date created

02th May 2020, **César Vila**

### Project Title:

#### The information behind COVID-19 Data

### Description

This project try to extract valuable information from **COVID figures** posted on the web.
In the link below I have writen some lines to summarize the analysis of the data:

https://link.medium.com/P42VrDcI65

### Motivation of the project

From the begining of the *pandemic*, we are daily checking Covid-19 data. However, the information given is tough to understand.
When you watch the news, you get different analysis and insights of the information, and they vary a lot.
I thought that it might be a correlation between coronavirus and population data in each country, so I wanted to make my own research to understand which information is behind of Covid-19 Data.
Population data used is:
  - People per sqkm
  - Migrants
  - Average Age
  - Urban population (%)
  - Country with Free Healthcare policies

### Summary of the result

I have obtained the following information:
 - **Migrations** is the variable  most related with COVID-19
 - the higher is the **Average Age** of the countries, the higher are the coronavirus cases
 - Countries with **free healthcare** policies have more cases, but it seems that is because they are making more tests

### Files used

The used files are: **covid.ipynb** as main script

The supporting scripts are: 
1. World Countries files (SHP) 
2. Pics created folder for the charts created

### Libraries used:
 - **numpy** and **pandas** to *create dataframes* and work with them
 - **requests** and from **bs4**, **BeautifulSoup** for *downloading data* from *worldometers* url
 - **matplotlib.pyplot** and **shapefile** for *plotting* great charts
 - **sklearn** for the *model* 

### Credits

Udacity
Medium
https://www.worldometers.info/coronavirus/
https://worldpopulationreview.com/countries/countries-with-universal-healthcare/
"World Countries". Downloaded from http://tapiquen-sig.jimdo.com. 
Carlos Efraín Porto Tapiquén. Orogénesis Soluciones Geográficas. Porlamar, Venezuela 2015.
Based on shapes from Enviromental Systems Research Institute (ESRI). Free Distribuition.

