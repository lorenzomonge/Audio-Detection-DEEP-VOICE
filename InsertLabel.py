# Open the CSV file
import csv

with open('extracted_features.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Check if the rows list is empty or has enough elements
if len(rows) == 0:
    print("The rows list is empty.")
elif len(rows) < 66:
    print("The rows list does not have enough elements.")
    # Add empty rows to the list until it has enough elements
    while len(rows) < 66:
        rows.append([])

    # Add the "LABEL" header as the last column
    rows[0].append("LABEL")

    # Set the label values for fake and real audio
    for i in range(0, 58):
        rows[i].append("FAKE")
    for i in range(58, 66):
        rows[i].append("REAL")

    # Write the modified data back to the CSV file
    with open('extracted_features.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
else:
    # The rows list has enough elements, no action needed
    pass
