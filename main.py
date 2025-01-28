import pandas as pd
from datetime import datetime

def process_csv(file_path, output_path):
    """
    Reads a CSV file, converts the 'Дата' column from format 'Friday, July 01 2012' to 'YYYY-MM-DD',
    and saves the modified file.

    Args:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed CSV file.
    """
    # Load the CSV file
    try:
        data = pd.read_csv(file_path, encoding='utf-8', delimiter=';')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='cp1251', delimiter=';')

    # Ensure the 'Дата' column exists
    if 'Дата' not in data.columns:
        raise ValueError("The column 'Дата' is not found in the CSV file.")

    # Convert 'Дата' column to format 'YYYY-MM-DD'
    def convert_date(date_str):
        try:
            return datetime.strptime(date_str, '%A, %B %d %Y').strftime('%Y-%m-%d')
        except ValueError:
            return date_str  # Return the original if parsing fails

    data['Дата'] = data['Дата'].apply(convert_date)

    # Save the processed data to a new file
    data.to_csv(output_path, index=False, encoding='utf-8', sep=';')

# Example usage
# Replace 'input.csv' with your file path and 'output.csv' with the desired output path
process_csv('input-2.csv', 'output.csv')
