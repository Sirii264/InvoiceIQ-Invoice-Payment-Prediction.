
import pandas as pd
import numpy as np
import os

# 1. Project paths
BASE = r"C:\Users\sridevi\Documents\InvoiceIQ"

RAW = os.path.join(BASE, "data", "raw", "test_invoices.csv")
PROCESSED = os.path.join(BASE, "data", "processed")
REPORTS = os.path.join(BASE, "reports")

os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

print("Starting InvoiceIQ Cleaning Pipeline...")


# 2. Load dataset
df = pd.read_csv(RAW)

print("Original Shape:", df.shape)

# 3. Audit original data
missing = df.isnull().sum().sum()
duplicates = df.duplicated().sum()

# 4. Convert dates
date_columns = [
    'invoice_date',
    'due_date',
    'payment_date'
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# 5. Standardise text
text_columns = df.select_dtypes(include='object').columns

for col in text_columns:
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# 6. Create payment features
paid = df['payment_status'].eq('Paid')

df['days_to_pay'] = np.nan
df['payment_delay'] = np.nan

valid_paid = paid & df['payment_date'].notna()

df.loc[valid_paid, 'days_to_pay'] = (
    df.loc[valid_paid, 'payment_date'] -
    df.loc[valid_paid, 'invoice_date']
).dt.days

df.loc[valid_paid, 'payment_delay'] = (
    df.loc[valid_paid, 'payment_date'] -
    df.loc[valid_paid, 'due_date']
).dt.days

df['paid_late'] = pd.Series(
    pd.NA, index=df.index, dtype='Int64'
)

df.loc[valid_paid, 'paid_late'] = (
    df.loc[valid_paid, 'payment_date'] >
    df.loc[valid_paid, 'due_date']
).astype('Int64')

# 7. Create invoice status
df['invoice_status'] = df['payment_status'].map({
    'Paid': 'Paid',
    'Pending': 'Open',
    'Overdue': 'Open',
    'Cancelled': 'Cancelled'
})

# 8. Create data dictionary
dictionary = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes.astype(str).values,
    'Missing Values': df.isnull().sum().values,
    'Unique Values': df.nunique().values
})

dictionary.to_csv(
    os.path.join(REPORTS, 'data_dictionary.csv'),
    index=False
)

# 9. Create column mapping
mapping = {
    'Invoice ID': 'invoice_id',
    'Customer': 'customer_id, customer_name',
    'Invoice Date': 'invoice_date',
    'Due Date': 'due_date',
    'Payment Date/Status': 'payment_date, payment_status',
    'Amount': 'total_amount',
    'Currency': 'currency',
    'Country': 'customer_country'
}

mapping_df = pd.DataFrame(
    list(mapping.items()),
    columns=['Project Field', 'Dataset Column']
)

mapping_df.to_csv(
    os.path.join(REPORTS, 'column_mapping.csv'),
    index=False
)

# 10. Save cleaned dataset
df.to_csv(
    os.path.join(PROCESSED, 'invoices_cleaned.csv'),
    index=False
)

# 11. Create validation summary
print("\nFINAL VALIDATION")

print("Final Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("Negative Amounts:", (df['total_amount'] < 0).sum())

print("\nInvoiceIQ Cleaning Completed Successfully!")
