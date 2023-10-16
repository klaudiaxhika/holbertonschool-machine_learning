Pandas is a popular open-source data analysis and manipulation library for Python. It provides data structures like Series and DataFrame for efficiently storing and analyzing data.

### What is a `pd.DataFrame`?

A `pd.DataFrame` is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns).
It is similar to a spreadsheet or a SQL table. You can create a DataFrame using various methods, including reading data from external sources
or creating one from scratch using dictionaries, lists, or other pandas Series.

#### How to create a `pd.DataFrame`:

**From a Dictionary:**
```python
import pandas as pd

data = {
    'Column1': [1, 2, 3],
    'Column2': ['A', 'B', 'C']
}

df = pd.DataFrame(data)
```

**From a List of Dictionaries:**
```python
data = [
    {'Column1': 1, 'Column2': 'A'},
    {'Column1': 2, 'Column2': 'B'},
    {'Column1': 3, 'Column2': 'C'}
]

df = pd.DataFrame(data)
```

### What is a `pd.Series`?

A `pd.Series` is a one-dimensional labeled array that can hold data of any type (integers, strings, floating-point numbers, Python objects, etc.).
It is the building block of a DataFrame.

#### How to create a `pd.Series`:

```python
import pandas as pd

data = [1, 2, 3, 4, 5]
series = pd.Series(data)
```

### How to load data from a file:

```python
import pandas as pd

# Reading data from a CSV file
df = pd.read_csv('file.csv')

# Reading data from an Excel file
df = pd.read_excel('file.xlsx')

# Reading data from a JSON file
df = pd.read_json('file.json')
```

### How to perform indexing on a `pd.DataFrame`:

#### Single Column Indexing:
```python
# Accessing a single column by name
df['Column1']

# Accessing multiple columns by names
df[['Column1', 'Column2']]
```

#### Row Indexing:
```python
# Accessing a row by index label
df.loc[0]

# Accessing a row by integer location
df.iloc[0]
```

### How to use hierarchical indexing with a `pd.DataFrame`:

```python
# Creating a DataFrame with hierarchical index
index = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2)], names=['Letter', 'Number'])
df = pd.DataFrame({'Data': [10, 20, 30, 40]}, index=index)
```

### How to slice a `pd.DataFrame`:

```python
# Slicing rows
df[1:3]

# Slicing columns
df.loc[:, 'Column1':'Column2']
```

### How to reassign columns:

```python
# Renaming columns
df.rename(columns={'Column1': 'NewColumn1', 'Column2': 'NewColumn2'}, inplace=True)
```

### How to sort a `pd.DataFrame`:

```python
# Sorting by values in a specific column
df.sort_values(by='Column1', ascending=False, inplace=True)

# Sorting by index
df.sort_index(level=['Column1', 'Column2'], ascending=[True, False], inplace=True)
```

### How to use boolean logic with a `pd.DataFrame`:

```python
# Using boolean conditions to filter the DataFrame
filtered_df = df[df['Column1'] > 2]

# Using multiple conditions
filtered_df = df[(df['Column1'] > 2) & (df['Column2'] == 'B')]
```

### How to merge/concatenate/join `pd.DataFrames`:

#### Concatenation:
```python
# Concatenating DataFrames vertically
result = pd.concat([df1, df2])

# Concatenating DataFrames horizontally
result = pd.concat([df1, df2], axis=1)
```

#### Merging:
```python
# Merging DataFrames based on a common column
merged_df = pd.merge(df1, df2, on='common_column')

# Merging on multiple columns
merged_df = pd.merge(df1, df2, on=['column1', 'column2'])
```

#### Joining:
```python
# Joining DataFrames on index
result = df1.join(df2, how='inner')

# Joining on a specific column
result = df1.join(df2, on='key_column', how='left')
```

### How to get statistical information from a `pd.DataFrame`:

```python
# Getting basic statistics for numerical columns
df.describe()

# Getting the mean of a specific column
mean_value = df['Column1'].mean()

# Getting the correlation matrix
correlation_matrix = df.corr()
```

### How to visualize a `pd.DataFrame`:

```python
import matplotlib.pyplot as plt

# Plotting a DataFrame
df.plot(x='Column1', y='Column2', kind='scatter')
plt.show()

# Creating a histogram
df['Column1'].plot(kind='hist')
plt.show()

# Creating a bar chart
df['Column2'].value_counts().plot(kind='bar')
plt.show()
```

Official pandas documentation: https://pandas.pydata.org/pandas-docs/stable/index.html
