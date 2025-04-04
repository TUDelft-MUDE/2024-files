import csv

# Read the CSV file and store data as a list of dictionaries
input_file = './content/GA_2_8/tickets/tickets.csv'
output_file = 'output.txt'

with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

# Write the data to a tab-separated file
with open(output_file, mode='w', encoding='utf-8') as outfile:
    # Write the header
    outfile.write('\t'.join(reader.fieldnames) + '\n')
    # Write the rows
    for row in data:
        outfile.write('\t'.join(row[field] for field in reader.fieldnames) + '\n')