# SKILL-ASSSESMENT-1
# AIM:
  To perform a comprehensive data science assessment using Python, applying key concepts such as data cleaning, analysis,
  and basic machine learning techniques.

# EXPLANATION:
Data Science involves extracting meaningful insights from structured or unstructured data using scientific methods, algorithms, and tools. Python, being a versatile programming language, provides libraries like Pandas, NumPy and  Scikit-learn that enable data manipulation, visualization, and modeling. This assessment ensures practical understanding and command over essential data science operations.

# ALGORITHM:
STEP 1: Import necessary libraries: pandas, numpy.

STEP 2: Create a dataframe with the help of pandas.

STEP 3: Perform basic data cleaning:

  - Handle missing values using .isnull(), .dropna(), or .fillna().

  - Remove duplicates using .duplicated().

  - Convert data types if necessary using .astype().

STEP 4: Perform data manipulation using pandas and numpy:

  - Filter rows or columns.

  - Apply aggregation functions (like .mean(), .sum(), etc.).

STEP 5: Display or save the results as required.

#  CODING AND OUTPUT:

NAME :  ROHIT GP

REGISTER NUMBER : 212224220082

```
# creating a dataFrame
import pandas as pd
df = pd.DataFrame(
    [[1,2,3],
     [4,5,6],
     [7,8,9]],
    index=[1,2,3],
    columns = ['a','b','c']
)
print(df)
```
![Screenshot 2025-05-11 104457](https://github.com/user-attachments/assets/882b79db-d92d-448d-a8e6-9105a2e1566c)

```
# creating dataframe from dictionary
import pandas as pd
mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}
myvar = pd.DataFrame(mydataset)
print(myvar)
```
![Screenshot 2025-05-11 104538](https://github.com/user-attachments/assets/fa1b7852-5c58-49f8-a9f9-01d575978343)

```
# Column addition
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Height': [5.5, 6.0, 5.8, 5.6],
    'Qualification': ['Engineer', 'Doctor', 'Artist', 'Teacher']
}

df = pd.DataFrame(data)

address = ['New York', 'Los Angeles', 'Chicago', 'Houston']
df['Address'] = address

print(df)
```
![Screenshot 2025-05-11 104600](https://github.com/user-attachments/assets/003b58e0-2026-446d-aac1-9dc177b20d81)

```
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [29, 34, 26, 31],
    'Address': ['New York', 'Chicago', 'Seattle', 'Austin'],
    'Qualification': ['Engineer', 'Lawyer', 'Designer', 'Analyst']
}

df = pd.DataFrame(data)

del df['Address']

print(df)
```
![Screenshot 2025-05-11 104614](https://github.com/user-attachments/assets/b31b3947-b56c-42ed-86e0-760fc234cfb3)

```
# column renaming
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [29, 34, 26, 31],
    'Address': ['New York', 'Chicago', 'Seattle', 'Austin'],
    'Qualification': ['Engineer', 'Lawyer', 'Designer', 'Analyst']
}

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

df.rename(columns={'Address': 'Place'}, inplace=True)

print("\nDataFrame after renaming column:\n", df)
```
![Screenshot 2025-05-11 104638](https://github.com/user-attachments/assets/9e948755-d96c-4394-9edd-b38e75d9332a)

```
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [29, 34, 26, 31],
    'Address': ['New York', 'Chicago', 'Seattle', 'Austin'],
    'Qualification': ['Engineer', 'Lawyer', 'Designer', 'Analyst']
}

df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

df.columns = ['A', 'B', 'C', 'D']
print("\nDataFrame after renaming columns:\n", df)
```
![Screenshot 2025-05-11 104703](https://github.com/user-attachments/assets/78148ffa-3fa1-451b-bd44-70d41e6c7326)

```
# Indexing and Selecting data
import pandas as pd
data = {
    'name': ['Emily', 'Frank', 'Grace', 'Henry'],
    'age': [28, 35, 22, 41],
    'gender': ['F', 'M', 'F', 'M'],
    'height': [1.68, 1.80, 1.60, 1.75]
}

df = pd.DataFrame(data)
df = df['name']

print(df)
```
![Screenshot 2025-05-11 104716](https://github.com/user-attachments/assets/cdad671f-6ae7-4936-8159-afd4ec091663)

```
# Multiple column selection
import pandas as pd
data = {
    'Name': ['Emily', 'Nathan', 'Sophia', 'Liam'],
    'Age': [28, 31, 26, 35],
    'Address': ['Boston', 'Denver', 'Atlanta', 'Phoenix'],
    'Qualification': ['MBA', 'B.Tech', 'MSc', 'PhD']
}

df = pd.DataFrame(data)

print(df[['Name', 'Qualification']])
```
![Screenshot 2025-05-11 104728](https://github.com/user-attachments/assets/28723622-e05e-45e3-8830-0157caff5391)

```
# Finding nlargest
data = {'name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000]}
df = pd.DataFrame(data)

top_salaries = df.nlargest(2, columns='salary')
print(top_salaries)
```
![Screenshot 2025-05-11 104740](https://github.com/user-attachments/assets/d2f0c2a1-9898-487e-8622-1ac4227a6340)

```
# Handling missing values using Dropna
import pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan'],
    'Age': [25, 32, None, 41, 28],
    'Salary': [50000, None, 70000, 90000, 60000]
}

df = pd.DataFrame(data)

df.dropna(inplace=True)

print(df)
```
![Screenshot 2025-05-11 104757](https://github.com/user-attachments/assets/c2550c1c-5d5c-4330-893f-f6f8c5572980)

```
# Filling missing values
import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan'],
    'Age': [25, np.nan, 35, 41, np.nan],
    'Salary': [50000, np.nan, 70000, np.nan, 60000]
}

df = pd.DataFrame(data)

df_filled = df.fillna(0)

print(df_filled)
```
![Screenshot 2025-05-11 104810](https://github.com/user-attachments/assets/d07cdd3e-a6c0-44bf-b26f-06ac92e16353)

```
# Sorting the dataframe with actual value
import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],
			'col3':['abc','def','ghi','xyz']})
df.sort_values(by='col2')
print(df)
```
![Screenshot 2025-05-11 104819](https://github.com/user-attachments/assets/e1fe6dcb-67d4-4946-8795-cce5f6a33bee)

```
#Sorting the dataframe with label
df = pd.DataFrame(np.random.randn(10,2),
		index=[1,4,6,2,3,5,9,8,0,7],columns = ['col2','col1'])
df
df1=df.sort_index(ascending=True)
print("Dataframe 1")
print(df1)
print("Dataframe 2")
df2=df.sort_index()
df2
```
![Screenshot 2025-05-11 104840](https://github.com/user-attachments/assets/b2a145ad-8e72-4b8b-b569-fa9b64a33958)

```
# Groupby commands
import pandas as pd
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}
df = pd.DataFrame(data)
df
```
![Screenshot 2025-05-11 104854](https://github.com/user-attachments/assets/27ea6e3e-1ea7-4b3e-ac54-78ad44653b0c)

```
print(df.groupby('Company')['Sales'].mean()) 
```
![Screenshot 2025-05-11 104904](https://github.com/user-attachments/assets/b95b3768-bb5f-408d-8a8f-b37cf7d3e6e7)

```
print(df.groupby('Company')['Sales'].min()) 
```
![Screenshot 2025-05-11 104914](https://github.com/user-attachments/assets/c0ee0698-4da1-479a-b3a2-60b594ca3c63)

```
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()
```
![Screenshot 2025-05-11 104924](https://github.com/user-attachments/assets/69d15990-3c00-4c54-af7a-77dc051110a1)

```
df.describe()
```
![Screenshot 2025-05-11 104935](https://github.com/user-attachments/assets/d9d2ae0a-a3c6-45ed-b111-6599e0843035)

```
# Checking for missing values
df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})
df
```
![Screenshot 2025-05-11 104948](https://github.com/user-attachments/assets/ddc049aa-5f32-4604-b80b-c8b3a3489dab)

```
df.isnull()
```
![Screenshot 2025-05-11 104957](https://github.com/user-attachments/assets/b9d256f2-368d-4a9e-b4a8-a8adec362199)

```
df.notnull()
```
![Screenshot 2025-05-11 105009](https://github.com/user-attachments/assets/936d07d0-316e-4efe-88d2-e5d009385fea)

```
df.dropna()
```
![Screenshot 2025-05-11 105018](https://github.com/user-attachments/assets/23840105-951d-4bd2-976e-c2a06cc3699e)

```
df.dropna(axis=1)
```
![Screenshot 2025-05-11 105029](https://github.com/user-attachments/assets/44eac2ee-7edb-4322-a918-3743e2774a1f)

```
#Replace NaN with a Scalar Value
df.fillna(value=10)
df['A'].fillna(value=df['A'].mean())
df
```
![Screenshot 2025-05-11 105055](https://github.com/user-attachments/assets/bd1a76a7-bd4d-4ee0-b5cd-79a7789fe56f)

```
df.fillna(method='ffill')
```
![Screenshot 2025-05-11 105109](https://github.com/user-attachments/assets/32e1c20c-fc8a-41c4-bf72-f9cf464cb651)

```
# Forward and backward filling
df = pd.DataFrame({'A':[1,np.nan,2],
                  'B':[5,np.nan,3],
                  'C':[1,2,3]})
df
```
![Screenshot 2025-05-11 105119](https://github.com/user-attachments/assets/c9277978-ec96-4f45-8bb1-39020e6c7f76)

```
# using backward filling
df.fillna(method='ffill')
```
![Screenshot 2025-05-11 105131](https://github.com/user-attachments/assets/f4b008b3-57f2-4fbc-b62e-d569d77a978e)

```
# using forward filling
df.fillna(method='bfill')
```
![Screenshot 2025-05-11 105143](https://github.com/user-attachments/assets/2202ec98-189c-4517-aa3c-428d75a2f850)

```
# DataFrame Commands
data = {
    'Name': ['John', 'Sarah', 'Mike', 'Emily', 'David'],
    'Age': [25, 31, 29, 35, 27],
    'Gender': ['M', 'F', 'M', 'F', 'M'],
    'Salary': [50000, 70000, 60000, 80000, 55000]
}
df = pd.DataFrame(data)
print("Head Values")
print(df.head(3))
print("Tail Values")
print(df.tail(3))
```
![Screenshot 2025-05-11 105155](https://github.com/user-attachments/assets/e3d55842-9e2f-496a-ade0-559b7665f6ea)

```
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
    'Age': [25, np.nan, 35, 41, np.nan],
    'Salary': [50000, np.nan, 70000, np.nan, 60000]
}

df = pd.DataFrame(data)

# This will check for non-duplicated values
~df.duplicated()
```
![Screenshot 2025-05-11 105206](https://github.com/user-attachments/assets/e93b2a02-9e2d-42ee-9c59-83c8f6336941)

```
# filtering in a dataframe
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Dave'],
    'age': [25, 32, 18, 47],
    'gender': ['F', 'M', 'M', 'M'],
    'height': [1.62, 1.78, 1.65, 1.83]
}

df = pd.DataFrame(data)

df_filtered = df[(df['gender'] == 'M') & (df['height'] > 1.7)]


print(df_filtered)
```
![Screenshot 2025-05-11 105217](https://github.com/user-attachments/assets/04eb6a00-ba29-4e38-bb81-f6363aab09dd)


# Result:
 Basic operations on the dataset were successfully performed using pandas and NumPy. The data was cleaned by handling missing values and removing duplicates.
