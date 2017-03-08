Title: Webscraping Indeed
Date: 2017-01-06
Category: Data Science
Tags: Data Science
Author: Avinash TAMBY

For this project, I scrape Indeed.com for data scientist salaries in cities across America and try to figure out whether or not a certain job post will pay above the median data scientist salary.


```python
# import everything I need

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler
```


```python
# Create list of cities I want to search Data Science jobs in in the format of what it looks like in the URL

cities = ['New+York,+NY', 'Los+Angeles,+CA', 'Chicago,+IL', 'Houston,+TX', 'Philadelphia,+PA', 'Phoenix,+AZ',
          'San+Antonio,+TX', 'San+Diego,+CA', 'Dallas,+TX', 'San+Jose,+CA', 'Austin,+TX', 'Jacksonville,+FL',
          'Steamtteaman+Francisco,+CA', 'Indianapolis,+IN', 'Columbus,+OH', 'Fort+Worth,+TX', 'Charlotte,+NC', 'Seattle,+WA',
          'Denver,+CO', 'El+Paso,+TX', 'Detroit,+MI', 'Washington,+DC', 'Boston,+MA', 'Memphis,+TN', 'Nashville,+TN', 
          'Portland,+OR', 'Oklahoma+City,+OK', 'Baltimore,+MD', 'Atlanta,+GA', 'Pittsburgh,+PA', 'Palo+Alto,+CA',
          'Mountain+View,+CA', 'Cupertino,+CA', 'Cambridge,+MA', 'Miami,+FL']
```

Below, I use BeautifulSoup to scrape Indeed. I take the job title, company name, location, summary and salary (if there is one listed). I'm going to find the median of all the jobs that *do* have a listed salary and then build a machine learning classifier to try to predict whether or not all of the *other* job posts will pay above the median.


```python
# Scrape

df = pd.DataFrame(columns=["title","company", "location","summary", "salary"])
p1 = "https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l="
p2 = "&fromage=any&limit=250&sort=&psf=advsrch"

count = 0

for j in cities:
    url = p1 + j + p2 
    html = urllib.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser', from_encoding="utf-8")
    print url

    title = []
    comp = []
    loc = []
    loc_code = []
    sal = []
    desc = []

    for i in soup.find_all(name='div', attrs={'class':' row result'}):
        title.append(i.find(name='h2').text)
        comp.append(i.find(name='span', attrs={'class':'company'}).text)
        loc.append(i.find(name='span', attrs={'class':'location'}).text)
        loc_code.append(count)
        try:
            sal.append(i.find(name='nobr').text)
        except:
            sal.append(None)
        desc.append(i.find(name='span', attrs={'class':'summary'}).text)

    title = map(lambda s: s.strip(), title)
    comp = map(lambda s: s.strip(), comp)
    loc = map(lambda s: s.strip(), loc)
    desc = map(lambda s: s.strip(), desc)

    jobs = pd.DataFrame(columns=["title","company", "location", "location_code", "summary", "salary"])
    jobs.title = title
    jobs.company = comp
    jobs.salary = sal
    jobs.location = loc
    jobs.location_code = loc_code
    jobs.summary = desc
    
    # Append df of jobs in each city to an overall all jobs df

    df = df.append(jobs)
    
    count += 1
```

    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=New+York,+NY&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Los+Angeles,+CA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Chicago,+IL&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Houston,+TX&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Philadelphia,+PA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Phoenix,+AZ&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=San+Antonio,+TX&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=San+Diego,+CA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Dallas,+TX&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=San+Jose,+CA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Austin,+TX&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Jacksonville,+FL&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Steamtteaman+Francisco,+CA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Indianapolis,+IN&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Columbus,+OH&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Fort+Worth,+TX&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Charlotte,+NC&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Seattle,+WA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Denver,+CO&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=El+Paso,+TX&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Detroit,+MI&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Washington,+DC&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Boston,+MA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Memphis,+TN&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Nashville,+TN&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Portland,+OR&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Oklahoma+City,+OK&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Baltimore,+MD&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Atlanta,+GA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Pittsburgh,+PA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Palo+Alto,+CA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Mountain+View,+CA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Cupertino,+CA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Cambridge,+MA&fromage=any&limit=250&sort=&psf=advsrch
    https://www.indeed.com/jobs?as_and=data+scientist&as_phr=&as_any=&as_not=&as_ttl=&as_cmp=&jt=fulltime&st=&salary=$70,000%2B&radius=25&l=Miami,+FL&fromage=any&limit=250&sort=&psf=advsrch



```python
df.drop_duplicates(inplace=True)
df.shape
```




    (2437, 6)



Cool, I was able to get 2,437 job posts. Now let's see how many of them have salaries missing.


```python
df.isnull().sum()
```




    company             0
    location            0
    location_code       0
    salary           2316
    summary             0
    title               0
    dtype: int64



Ooh, 2,316 missing salaries.


```python
notnull = df[df.salary.notnull()]
```

I'm going to clean up the salary figure and only take the lower end of the range.


```python
df2 = notnull[(~notnull.salary.str.contains('hour')) & (~notnull.salary.str.contains('month'))]
```


```python
df2.salary = df2.salary.apply(lambda x: x.replace(' a year',''))
df2.salary = df2.salary.apply(lambda x: x.replace('$',''))
df2.salary = df2.salary.apply(lambda x: x.replace(',',''))
df2.salary = df2.salary.apply(lambda x: x.split('-')[0])
df2.salary = df2.salary.astype(float)
```

    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/pandas/core/generic.py:2701: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[name] = value



```python
df2.shape
```




    (107, 6)



Yeah, again. I only have 107 salaries for 2400 job posts. But, we shall continue. Let's find the median. And create a binary target: 1 if above, 0 if equal or below the median.


```python
med = df2.salary.median()
```


```python
df2['target'] = np.where(df2.salary > med, 1, 0)
df2.head()
```

    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company</th>
      <th>location</th>
      <th>location_code</th>
      <th>salary</th>
      <th>summary</th>
      <th>title</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Oliver James Associates</td>
      <td>New York, NY</td>
      <td>0.0</td>
      <td>90000.0</td>
      <td>Identifies and develops data sources to solve ...</td>
      <td>Data Scientist</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Venturi Ltd</td>
      <td>New York, NY</td>
      <td>0.0</td>
      <td>200000.0</td>
      <td>Data Scientist, FinTech, Python, R, Machine Le...</td>
      <td>Data Scientist ( FinTech / Python / R / Machin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Selby Jennings</td>
      <td>New York, NY 10167 (Midtown area)</td>
      <td>0.0</td>
      <td>150000.0</td>
      <td>Mentored junior data scientists. Data Scientis...</td>
      <td>Data Scientist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Selby Jennings</td>
      <td>New York, NY 10167 (Midtown area)</td>
      <td>0.0</td>
      <td>160000.0</td>
      <td>Data Scientist | New York, NY. A pioneering In...</td>
      <td>Data Scientist | Investment Research</td>
      <td>1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Beeswax</td>
      <td>New York, NY</td>
      <td>0.0</td>
      <td>130000.0</td>
      <td>A minimum of 5 years experience in Machine lea...</td>
      <td>Director / VP Data Science</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



I'm going to save this data to a csv.


```python
df2.to_csv('df2.csv', encoding='utf-8')
```


```python
jobs = pd.read_csv('df2.csv')
jobs.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>company</th>
      <th>location</th>
      <th>location_code</th>
      <th>salary</th>
      <th>summary</th>
      <th>title</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Oliver James Associates</td>
      <td>New York, NY</td>
      <td>0.0</td>
      <td>90000.0</td>
      <td>Identifies and develops data sources to solve ...</td>
      <td>Data Scientist</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>Venturi Ltd</td>
      <td>New York, NY</td>
      <td>0.0</td>
      <td>200000.0</td>
      <td>Data Scientist, FinTech, Python, R, Machine Le...</td>
      <td>Data Scientist ( FinTech / Python / R / Machin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36</td>
      <td>Selby Jennings</td>
      <td>New York, NY 10167 (Midtown area)</td>
      <td>0.0</td>
      <td>150000.0</td>
      <td>Mentored junior data scientists. Data Scientis...</td>
      <td>Data Scientist</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Selby Jennings</td>
      <td>New York, NY 10167 (Midtown area)</td>
      <td>0.0</td>
      <td>160000.0</td>
      <td>Data Scientist | New York, NY. A pioneering In...</td>
      <td>Data Scientist | Investment Research</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65</td>
      <td>Beeswax</td>
      <td>New York, NY</td>
      <td>0.0</td>
      <td>130000.0</td>
      <td>A minimum of 5 years experience in Machine lea...</td>
      <td>Director / VP Data Science</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



I'm just going to take a minute to step back and remember something regarding the baseline accuracy.
The baseline accuracy is 50% because we are basing the classification on the median. Thus, by definition, 50% of the data is 1 and 50% is 0. Thus, if we classified everything as 1 or everything as 0, we would have an accuracy of 50%.


```python
def classifier(X, y, clf):
    
    X = StandardScaler().fit_transform(X)

    acc = []
    for i in range(50):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc.append(accuracy_score(y_pred, y_test))
    
    return np.mean(acc)
```


```python
rf = RandomForestClassifier()

X = jobs.drop(['Unnamed: 0', 'company', 'location', 'salary', 'summary', 'title', 'target'], axis=1)
X = StandardScaler().fit_transform(X)
y = jobs.target

classifier(X, y, rf)
```




    0.67636363636363628



I run my classifier (in the case above, a Random Forest) 50x to adjust for fluctuations in accuracy, and I get about 67% accuracy. That's ok, but I can maybe do better. I'm going to some feature engineering and find more senior-level posts (if the job title has some senior-level word like lead or principal in it).


```python
df2['senior'] = 0

for i in range(len(df2.title)):
    if ('Chief' in df2.title.iloc[i]):
        df2['senior'].iloc[i] = 1
        
    elif ('Manager' in df2.title.iloc[i]):
        df2['senior'].iloc[i] = 1
        
    elif ('Senior' in df2.title.iloc[i]):
        df2['senior'].iloc[i] = 1
        
    elif ('Director' in df2.title.iloc[i]):
        df2['senior'].iloc[i] = 1
    
    elif ('Sr' in df2.title.iloc[i]):
        df2['senior'].iloc[i] = 1
    
    elif ('Lead' in df2.title.iloc[i]):
        df2['senior'].iloc[i] = 1
        
    elif ('Principal' in df2.title.iloc[i]):
        df2['senior'].iloc[i] = 1
```

    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':
    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/pandas/core/indexing.py:132: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._setitem_with_indexer(indexer, value)
    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:17: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:20: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    /Users/avinashtamby/anaconda/envs/py27/lib/python2.7/site-packages/ipykernel/__main__.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy



```python
df2.to_csv('df2.csv', encoding='utf-8')
jobs = pd.read_csv('df2.csv')
```


```python
X = jobs.drop(['Unnamed: 0', 'company', 'location', 'salary', 'summary', 'title', 'target'], axis=1)

classifier(X, y, rf)
```




    0.68272727272727263



This adds some value, but the accuracy still isn't great.


```python
tvec = TfidfVectorizer(stop_words='english')
tvec.fit(jobs.summary)
tvec  = pd.DataFrame(tvec.transform(jobs.summary).todense(), columns=tvec.get_feature_names())

classifier(tvec, y, rf)
```




    0.69636363636363641



Just barely better, but I'll take it.


```python
df3 = pd.concat([X, tvec], axis=1)
acc = []

for i in range(50):

    clf = RandomForestClassifier()

    X_train, X_test, y_train, y_test = train_test_split(df3, y, test_size = 0.2)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc.append(accuracy_score(y_pred, y_test))

print np.mean(acc)

imp =  pd.DataFrame(clf.feature_importances_, index=df3.columns, columns=['importance'])
imp.head()
print imp.sort_values(['importance'], ascending=False).head()
```

    0.730909090909
                   importance
    location_code    0.052979
    team             0.042358
    analytics        0.032552
    scientists       0.031191
    data             0.030282


The words and features above have the most predictive value when classifying jobs at above or below the median. Location makes sense, I imagine more expensive cities pay more. It's weird that 'data' is in there as these should all be data science jobs. But generally, it seems that quantitative aspects like 'analytics' and 'scientists' are pretty predictive.

Next, I take a look at the confusion matrix, precision and recall.


```python
print confusion_matrix(y_test, y_pred)
print classification_report(y_test, y_pred)
```

    [[11  0]
     [ 2  9]]
                 precision    recall  f1-score   support
    
              0       0.85      1.00      0.92        11
              1       1.00      0.82      0.90        11
    
    avg / total       0.92      0.91      0.91        22
    



```python
acc = []

for i in range(50):

    clf = GradientBoostingClassifier()

    X_train, X_test, y_train, y_test = train_test_split(df3, y, test_size = 0.2)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc.append(accuracy_score(y_pred, y_test))

print np.mean(acc)

imp =  pd.DataFrame(clf.feature_importances_, index=df3.columns, columns=['importance'])
imp.head()
print imp.sort_values(['importance'], ascending=False).head()
```

    0.675454545455
                importance
    learning      0.076278
    processing    0.070643
    analytics     0.057618
    data          0.054082
    developer     0.049816


So, although running a GB classifier doesn't perform as well as the other classifiers I've tried, I think the word importances are pretty key here. 'Learning' was the most important feature, so I imagine employers really value people who are willing to learn.
