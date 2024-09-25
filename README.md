## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:

  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:


#1. FUNCTION TRANSFORMATION:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/8d3d821e-89e1-4709-8467-aba553ea4485)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/27b0b23f-61e7-4a2d-b37e-18c0cc40f658)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c2c220c6-47b5-42b4-9141-5cd19193b9ac)

```

df.skew()
```
![image](https://github.com/user-attachments/assets/e7585703-e4af-4629-a957-357d47d759a2)
```
df["Highly Positive Skew"]=np.log(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/d9379076-3aa7-4085-bd5a-34acc16dd6e3)
```
df["Moderate Positive Skew"]=np.reciprocal(df["Moderate Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/53cccd5d-9218-4ce2-aa43-0e4bc4166371)
```
df["Highly Negative Skew"]=np.sqrt(df["Highly Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/0d72efd6-2381-46de-8b58-bc20643a9c02)

```
df["Highly Positive Skew"]=np.square(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/b7af63b3-6f08-44f5-b79f-c61cec8501e1)
```
df.skew( )
```
![image](https://github.com/user-attachments/assets/d5f0b601-91dc-4de9-97ef-fc8b9594ab4b)
```
df["Highly Positive Skew_boxcox"],parameter=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/56c495ea-ee73-42c7-a51e-52757bebe289)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/c35fa821-d9e4-4c6b-a902-5286a8000c3c)

#POWER TRANSFORMATION
```
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/30d8e52a-9ed1-4d11-85e3-219ffe60d967)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/56f78389-ffaf-46d3-91b0-c95ca2928560)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9074d7d3-0db4-4d9d-afbd-fc5066eee23f)

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
