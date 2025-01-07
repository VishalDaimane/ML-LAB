import pandas as pd

def find_s(data):
    # Initialize the most specific hypothesis
    hypothesis = ['0'] * (len(data.columns) - 1)

    for _, row in data.iterrows():
        if row.iloc[-1] == 1:  # Check for positive examples using the numeric value
            for i in range(len(hypothesis)):
                hypothesis[i] = row.[i] if hypothesis[i] in ['0', row[i]] else '?'

    return hypothesis

# Load dataset
data = pd.read_csv("ENJOYSPORT.csv")

print("Training Data:")
print(data)

# Run the Find-S algorithm
hypothesis = find_s(data)
print("\nMost Specific Hypothesis:", hypothesis)
