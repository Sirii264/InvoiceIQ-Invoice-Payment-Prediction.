
# InvoiceIQ - Week 3 Feature Engineering

## Project Objective
Prepare invoice data for machine learning by creating
predictive features and target variables.

## Feature Engineering
- Invoice month, quarter and weekday
- Month-end indicator
- Payment term days
- Previous customer invoice count
- Historical average invoice amount
- Previous average payment delay
- Previous late-payment rate
- Log-transformed invoice amount
- Invoice amount category
- Amount compared with customer historical average

## Target Variables
- paid_late_target: Classification target
- delay_days: Regression target

Targets are created only for invoices with confirmed
payment outcomes.

## Data Splitting
Chronological split:
- Training: 70%
- Validation: 15%
- Testing: 15%

## Preprocessing
- Missing value imputation
- One-hot encoding
- Standard scaling

Preprocessing is fitted on training data only.

## Important Limitation
The dataset contains zero late-payment examples.
A meaningful late-payment classification model cannot
be trained using the current target distribution.

## Output
Model-ready feature matrices, target datasets,
feature catalogues and leakage documentation.
