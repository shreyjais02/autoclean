"""
Generates a realistic dirty/messy dataset to test the preprocessing pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_dirty_dataset(n_rows: int = 500, output_path: str = "data/input/raw_customers.csv") -> Path:
    np.random.seed(42)

    # 1. IDs
    ids = [f"USR_{1000 + i}" for i in range(n_rows)]

    # 2. Age (with missing values and outliers)
    ages = np.random.normal(35, 10, n_rows).round().astype(object)
    # inject outliers
    ages[5] = 250
    ages[12] = -15
    # inject missing
    ages[np.random.choice(n_rows, size=30, replace=False)] = np.nan

    # 3. Income formatted as currency strings ($45,000.50)
    incomes_num = np.random.exponential(scale=50000, size=n_rows) + 20000
    incomes = [f"${val:,.2f}" for val in incomes_num]
    # inject extreme outlier
    incomes[10] = "$1,500,000.00"
    # inject missing
    for idx in np.random.choice(n_rows, size=25, replace=False):
        incomes[idx] = np.nan

    # 4. Dates
    start_date = pd.Timestamp("2022-01-01")
    dates = [
        (start_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d %H:%M:%S")
        for d in np.random.randint(0, 700, size=n_rows)
    ]

    # 5. Categorical column: Subscription Tier (low cardinality)
    tiers = np.random.choice(["Basic", "Pro", "Enterprise", None], size=n_rows, p=[0.5, 0.3, 0.15, 0.05])

    # 6. Categorical column: Country (medium cardinality)
    countries = np.random.choice(["USA", "UK", "Canada", "Germany", "France", "Japan", "Australia", None], size=n_rows, p=[0.4, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])

    # 7. Constant column (should be dropped)
    constant_col = ["RetailApp"] * n_rows

    # 8. High-missing column (95% null, should be dropped)
    fax_numbers = [f"555-01{i:02d}" if np.random.rand() < 0.05 else None for i in range(n_rows)]

    # 9. Target variable
    churn = np.random.choice([0, 1], size=n_rows, p=[0.75, 0.25])

    df = pd.DataFrame({
        "customer_id": ids,
        "signup_date": dates,
        "age": ages,
        "annual_income": incomes,
        "plan_tier": tiers,
        "country": countries,
        "system_source": constant_col,
        "fax_number": fax_numbers,
        "churn": churn,
    })

    # Add duplicate rows
    duplicates = df.sample(n=20, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Generated raw dirty dataset with {len(df)} rows and {df.shape[1]} columns -> {out_file.resolve()}")
    return out_file


if __name__ == "__main__":
    generate_dirty_dataset()
