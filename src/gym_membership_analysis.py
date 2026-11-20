#!/usr/bin/env python3

# Gym & Fitness Center Membership Analytics
# Name: MO SAJID |||  Addmission NO. : 2026B01010277

# Github Repo Link : https://github.com/mosajid782-stuward/Gym-Fitness-Membership-Analytics
# Run from the project root OR from this notebook. The CSV path is resolved automatically.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_dataset():
    candidates = [
        Path("data/12_gym_membership.csv"),
        Path("../data/12_gym_membership.csv"),
        Path("12_gym_membership.csv"),
        Path("../12_gym_membership.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Keep 12_gym_membership.csv in the project's data/ folder."
    )

DATA_PATH = find_dataset()
df = pd.read_csv(DATA_PATH)

print("TASK 1 — Load and inspect dataset")
print("Shape:", df.shape)
print(df.head())
print(df.info())

# Task 2 — Check missing and duplicate Checkin_ID values
print("\nTASK 2 — Checkin_ID quality")
print("Missing Checkin_ID:", df["Checkin_ID"].isna().sum())
print("Duplicate Checkin_ID rows:", df["Checkin_ID"].duplicated().sum())

# Remove duplicate check-in IDs, keeping the first occurrence
gym_clean = df.drop_duplicates(subset="Checkin_ID", keep="first").copy()
print("Rows after duplicate removal:", len(gym_clean))

print("\nTASK 3 — Clean missing workout duration")
# Task 3 — Clean missing workout duration using median
gym_clean["Workout_Duration_Mins"] = pd.to_numeric(
    gym_clean["Workout_Duration_Mins"], errors="coerce"
)
duration_median = gym_clean["Workout_Duration_Mins"].median()
missing_duration_before = gym_clean["Workout_Duration_Mins"].isna().sum()
gym_clean["Workout_Duration_Mins"] = gym_clean["Workout_Duration_Mins"].fillna(duration_median)
print("Missing workout durations imputed:", missing_duration_before)
print("Median used:", duration_median)

print("\nTASK 4 — Parse check-in time and derive weekday/hour")
# Task 4 — Parse check-in time and derive weekday/hour
gym_clean["Checkin_Time"] = pd.to_datetime(gym_clean["Checkin_Time"], errors="coerce")
invalid_time_rows = gym_clean["Checkin_Time"].isna().sum()
gym_clean = gym_clean.dropna(subset=["Checkin_Time"]).copy()
gym_clean["Weekday"] = gym_clean["Checkin_Time"].dt.day_name()
gym_clean["Hour"] = gym_clean["Checkin_Time"].dt.hour
gym_clean["Month"] = gym_clean["Checkin_Time"].dt.to_period("M").astype(str)
print("Invalid timestamps removed:", invalid_time_rows)
print(gym_clean[["Checkin_Time", "Weekday", "Hour", "Month"]].head())

# Task 5 — Total check-ins and average duration
print("\nTASK 5 — Overall check-in metrics")
total_checkins = len(gym_clean)
average_duration = gym_clean["Workout_Duration_Mins"].mean()
print("Total check-ins:", total_checkins)
print("Average workout duration (minutes):", round(average_duration, 2))

# Task 6 — Compare membership tiers
tier_summary = gym_clean.groupby("Membership_Tier").agg(
    Checkins=("Checkin_ID", "count"),
    Unique_Members=("Member_ID", "nunique"),
    Avg_Duration_Mins=("Workout_Duration_Mins", "mean")
).sort_values("Checkins", ascending=False)
print("\nTASK 6 — Membership tier comparison")
print(tier_summary.round(2))

# Task 7 — Peak check-in hours and busiest days
hour_summary = gym_clean["Hour"].value_counts().sort_index()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_summary = gym_clean["Weekday"].value_counts().reindex(weekday_order, fill_value=0)
print("\nTASK 7 — Peak hours and busiest days")
print("Busiest hour(s):", hour_summary[hour_summary == hour_summary.max()].index.tolist())
print("Busiest day(s):", weekday_summary[weekday_summary == weekday_summary.max()].index.tolist())
print(hour_summary.rename("Checkins").to_frame())
print(weekday_summary.rename("Checkins").to_frame())

# Task 8 — Fitness class attendance ranking
class_summary = gym_clean["Fitness_Class"].fillna("Not Specified").value_counts()
print("\nTASK 8 — Fitness class attendance")
print(class_summary.rename("Checkins").to_frame())

# Task 9 — Gender-wise workout duration comparison
gender_summary = gym_clean.groupby("Gender")["Workout_Duration_Mins"].agg(
    ["count", "mean", "median"]
).round(2)
print("\nTASK 9 — Gender-wise workout duration")
print(gender_summary)

# Task 10 — Estimate monthly revenue by membership tier.
# Monthly_Fee is repeated on each check-in row. To avoid multiplying a member's fee
# by their number of visits, estimate revenue using unique Member_ID + Month + Tier.
# If a member has more than one fee in a month/tier, use the mean fee for that group.
member_month_tier = gym_clean.groupby(
    ["Member_ID", "Month", "Membership_Tier"], as_index=False
).agg(Monthly_Fee=("Monthly_Fee", "mean"))
revenue_summary = member_month_tier.groupby(
    ["Month", "Membership_Tier"]
)["Monthly_Fee"].sum().unstack(fill_value=0)
print("\nTASK 10 — Estimated monthly revenue by tier")
print(revenue_summary.round(2))

# Task 11 — Low-engagement members: fewer than 2 check-ins per member-month
member_month_visits = gym_clean.groupby(
    ["Member_ID", "Month"]
).size().rename("Visits").reset_index()
low_engagement = member_month_visits[member_month_visits["Visits"] < 2].copy()
print("\nTASK 11 — Low-engagement member-month records")
print("Low-engagement member-month records:", len(low_engagement))
print("Distinct members with at least one low-engagement month:", low_engagement["Member_ID"].nunique())
print(low_engagement.head(10))

# Task 12 — NumPy median, variance, and quartiles
durations = gym_clean["Workout_Duration_Mins"].to_numpy(dtype=float)
np_median = np.median(durations)
np_variance = np.var(durations)  # population variance (ddof=0)
q1, q2, q3 = np.percentile(durations, [25, 50, 75])
print("\nTASK 12 — NumPy statistics")
print(f"Median: {np_median:.2f}")
print(f"Population variance: {np_variance:.2f}")
print(f"Q1: {q1:.2f}, Q2: {q2:.2f}, Q3: {q3:.2f}")

# Task 13 — IQR outlier detection
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = gym_clean[
    (gym_clean["Workout_Duration_Mins"] < lower_bound) |
    (gym_clean["Workout_Duration_Mins"] > upper_bound)
]
print("\nTASK 13 — IQR outliers")
print(f"IQR: {iqr:.2f}; lower bound: {lower_bound:.2f}; upper bound: {upper_bound:.2f}")
print("Outlier rows:", len(outliers))
print(outliers.head())

# Task 14 — Workout intensity labels using duration bands
# These are simple project-defined duration categories, not clinical intensity measures.
gym_clean["Workout_Intensity"] = pd.cut(
    gym_clean["Workout_Duration_Mins"],
    bins=[-np.inf, 30, 60, 90, np.inf],
    labels=["Short", "Moderate", "Long", "Extended"]
)
print("\nTASK 14 — Workout duration categories")
print(gym_clean["Workout_Intensity"].value_counts().to_frame("Checkins"))

# Visual summaries
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
hour_summary.plot(kind="bar", ax=axes[0], title="Check-ins by Hour")
weekday_summary.plot(kind="bar", ax=axes[1], title="Check-ins by Weekday")
plt.tight_layout()
plt.show()

class_summary.plot(kind="bar", figsize=(8, 4), title="Fitness Class Attendance")
plt.ylabel("Check-ins")
plt.tight_layout()
plt.show()

revenue_summary.plot(kind="bar", figsize=(10, 5), title="Estimated Monthly Revenue by Tier")
plt.ylabel("Estimated revenue (currency units)")
plt.tight_layout()
plt.show()

# Task 15 — Five recommendations
print("\nTASK 15 — Business recommendations")
recommendations = [
    "Schedule more trainer and equipment coverage during the highest check-in hours.",
    "Use Saturday attendance patterns to plan staffing, cleaning, and class capacity.",
    "Promote lower-attendance classes with trial sessions and member feedback.",
    "Contact members with fewer than two visits in a month and offer a simple re-engagement plan.",
    "Review membership-tier pricing and monthly-fee records regularly; revenue here is an estimate."
]
for number, recommendation in enumerate(recommendations, start=1):
    print(f"{number}. {recommendation}")

# Optional: save the cleaned analysis dataset beside the notebook output
output_path = Path("gym_membership_cleaned.csv")
gym_clean.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path.resolve()}")
