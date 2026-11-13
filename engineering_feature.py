import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
train = pd.read_csv("data/train.csv")
holidays = pd.read_csv("data/holidays_events.csv")

# Convert date columns
train["date"] = pd.to_datetime(train["date"])
holidays["date"] = pd.to_datetime(holidays["date"])

# Create holiday indicator
holiday_dates = holidays["date"].drop_duplicates()

train["is_holiday"] = train["date"].isin(
    holiday_dates
).astype(int)

# Select one store and product family
df = train[
    (train["store_nbr"] == 1) &
    (train["family"] == "GROCERY I")
].copy()

df = df.sort_values("date")

print(df.head())

print("\nHoliday count:")
print(df["is_holiday"].value_counts())

print("\nPromotion statistics:")
print(df["onpromotion"].describe())