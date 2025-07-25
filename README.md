# Analysis on BMW Used Cars for Company’s Fleet

## Problem Background

The company is tasked with selecting 5 pre-owned BMW cars for its fleet, with a budget of  **£150,000**. The goal is to buy 3 **NON-EV** units, 1 **Hybrid** unit, and 1 **Electric** unit, focusing on minimizing depreciation and maximizing the vehicle's residual value. The cars will be categorized into three price ranges:  **£15,000 (low)**,  **£30,000 (medium)**, and  **£45,000 (high)**. All vehicles will be purchased regardless of their price category.

## Project Output

The company has determined the selection of 5 operational vehicles with the following composition:

1. **3 Units of BMW NON-EV Cars**
2. **1 Unit of BMW Hybrid Car**
3. **1 Unit of BMW Electric Car**

## Dataset Overview

The dataset we're using comes from Kaggle. The data is generally clean, but some columns lack sufficient data, which significantly affects the statistical processing results. The dataset contains information about BMW cars, including various attributes such as model, year, price, and fuel type. Below is a summary of the columns and their respective data types and non-null counts:

| **Column Name** | Data Type | **Description**                                                 |
| --------------------- | --------- | --------------------------------------------------------------------- |
| `model`             | object    | Unique identifier for each car model.                                 |
| `year`              | int64     | Year of manufacture of the car.                                       |
| `price`             | int64     | Price of the car in the dataset.                                      |
| `transmission`      | object    | Type of car transmission (e.g., automatic, manual).                   |
| `mileage`           | int64     | Mileage of the car, representing how many miles the car has traveled. |
| `fuelType`          | object    | Type of fuel used by the car (e.g., petrol, diesel, electric).        |
| `tax`               | int64     | Tax amount associated with the car.                                   |
| `mpg`               | float64   | Miles per gallon, representing the fuel efficiency of the car.        |
| `engineSize`        | float64   | Size of the engine in liters.                                         |

* The dataset contains **10,781 rows** and  **9 columns** .
* **No Missing Values** : All columns are free from null values.
* **Duplicated Data** : Duplicates are present in categorical features such as car model and year.

## Method

The methods used in the analysis include central tendency calculations, descriptive statistics using confidence intervals to understand the potential profits, and inferential statistics using Pearson correlation to examine the relationships between variables.

## Stacks

The analysis is performed using Python to manage the data and conduct statistical calculations. Tableau is used for data visualization. In Python, I utilized the **pandas** library for data management, **scipy** for statistical analysis, and **seaborn** and **matplotlib** for data visualization.

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## References

- [BMW Used Car Dataset from Kaggle](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data?select=bmw.csv)
- [Tableau Visualization](https://public.tableau.com/app/profile/rafi.siregar/viz/milestone_17478445912520/Dashboard2?publish=yes)
