import pandas as pd
import numpy as np
import re
import csv
import logging
import multiprocessing
from multiprocessing import Pool
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_chunk(chunk_data):
    """Process a chunk of data and return DataFrame."""
    try:
        chunk_id, df_chunk, original_columns = chunk_data
        logger.info(f"Processing chunk {chunk_id} +++")
        logger.info(f"Chunk {chunk_id} columns: {df_chunk.columns.tolist()} +++")
        
        # Extract information

        product_col = 'Номенклатура Товарів Послуг'
        logger.info(f"PRODUCT_COL {product_col, df_chunk[product_col]} +++")
        quantity_col = "Кількість об’єм  Обсяг"  # Using actual column name
        price_col = 'Ціна Без ПДВ'
        logger.info(f"PRICE_COL {price_col, df_chunk[price_col]} +++")
        unit_col = 'Одиниця Виміру Товару  Послуги'
        df_chunk[price_col] = pd.to_numeric(df_chunk[price_col], errors="coerce")
        m = df_chunk
        # Create products catalog for this chunk by grouping unique products
        logger.info(f"Chunk {chunk_id} before groupby - columns: {df_chunk.columns.tolist()} +++")
        products_catalog = df_chunk.groupby(product_col).agg({
            unit_col: 'first',
            quantity_col: 'sum',
            price_col: 'mean'
        }).reset_index()
        logger.info(f"Chunk {chunk_id} after groupby - columns: {products_catalog.columns.tolist()} +++")
        logger.info(f"Products catalog {chunk_id} preview: {products_catalog.head()} @@@@@@@@@@@@@@@@@@")
        n = df_chunk
        # Add extracted information
        products_catalog['brand'] = products_catalog[product_col].apply(extract_brand)
        products_catalog['volume_liters'] = products_catalog[product_col].apply(
            extract_volume
        ).fillna(0)
        products_catalog['is_lubricant'] = products_catalog[product_col].apply(
            is_lubricant_related
        )
        
        # Create result DataFrame with original column names
        result_df = pd.DataFrame(columns=original_columns)
        logger.info(f"Chunk {chunk_id} result_df columns: {result_df.columns.tolist()} +++")
        
        for idx, row in enumerate(products_catalog.itertuples(), 1): #(8.12) row - tuple - get data as from tuple -through indexation
            result_df.loc[idx] = [
                idx,  # н п.п.
                pd.Timestamp.now().strftime('%Y-%m-%d'),  # Дата
                row[products_catalog.columns.get_loc(product_col)],  # Номенклатура Товарів Послуг
                row[products_catalog.columns.get_loc(unit_col)],  # Одиниця Виміру
                row[products_catalog.columns.get_loc(quantity_col)],  # Use original quantity value
                0,  # Сума Без ПДВ
                row[products_catalog.columns.get_loc(price_col)]  # Ціна Без ПДВ
            ]
        
        logger.info(f"Completed processing chunk {chunk_id} - final columns: {result_df.columns.tolist()} +++")
        logger.info(f"Sample of result_df: {result_df.head()} +++")
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {str(e)} +++")
        logger.error(f"Original columns were: {original_columns} +++")
        return pd.DataFrame(columns=original_columns)


def extract_volume(text):
    """Extract volume from product name."""
    try:
        volume_patterns = {
            'ml': r'(\d+)\s*[мm][лl]',
            'l': r'(\d+)\s*[лl]',
            'g': r'(\d+)\s*[гg][рr]',
            'kg': r'(\d+)\s*[кk][гg]'
        }
        
        for unit, pattern in volume_patterns.items():
            match = re.search(pattern, str(text), re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if unit == 'ml':
                    return value / 1000  # convert to liters
                elif unit == 'l':
                    return value
                elif unit == 'g':
                    return value / 1000  # approximate conversion to liters
                elif unit == 'kg':
                    return value  # approximate conversion to liters
        return None
    except Exception as e:
        logger.error(f"Error extracting volume from '{text}': {str(e)}")
        return None


def is_lubricant_related(product_name):
    """Check if product is related to lubricant business."""
    try:
        lubricant_keywords = [
            'масло', "мастило", "олива", "Мастильно-холодильна рідина", "рідина мастильно-холодильна",
            'смазка', 'жидкость', 'oil', 'fluid', 'grease', 'лубрикант', 'очиститель', 'cleaner', 'присадка', 'additive'
        ]
        product_lower = str(product_name).lower()
        # Log the product name being checked
        logger.debug(f"Checking product: '{product_name}' (lowercase: '{product_lower}') %%%%")
        
        for keyword in lubricant_keywords:
            if keyword.lower() in product_lower:
                logger.debug(f"Match found: '{keyword}' in '{product_name}' %%%%")
                return True
        logger.debug(f"No lubricant keywords found in '{product_name}' %%%%")
        return False
    except Exception as e:
        logger.error(
            f"Error checking lubricant relation for '{product_name}': {str(e)} %%%%",
            exc_info=True
        )
        return False


def extract_brand(product_name):
    """Extract brand from product name."""
    try:
        # Look for text in quotes or before first space
        brand_match = re.search(
            r'"([^"]+)"|\{([^}]+)\}|^([^\s]+)', 
            str(product_name)
        )
        if brand_match:
            return next(
                group for group in brand_match.groups() if group is not None
            )
        return 'Unknown Brand'
    except Exception as e:
        logger.error(f"Error extracting brand from '{product_name}': {str(e)}")
        return 'Unknown Brand'


def main():
    try:
        logger.info("Starting data processing... +++")

        # Read the CSV file
        logger.info("Reading input CSV file... +++")
        try:
            df = pd.read_csv('output_test.csv', sep=';', encoding='utf-8')
            logger.info("Successfully read CSV with UTF-8 encoding +++")
            # Логируем первую строку и колонки DataFrame
            logger.info(f"DataFrame head after reading: {df.head()} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")  # Логирование первых 5 строк
            logger.info(f"DataFrame columns: {df.columns.tolist()} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")  # Логирование всех колонок

            # logger.info(f"Chunk {chunk_id} head before processing: {df_chunk.head()} @@@@@@@@@@@@@@@@@@@@@@@@")
        except UnicodeDecodeError:
            logger.info("UTF-8 encoding failed, trying CP1251... +++")
            df = pd.read_csv('output_test.csv', sep=';', encoding='cp1251')
            logger.info("Successfully read CSV with CP1251 encoding +++")

        # Log initial data info
        total_rows = len(df)
        logger.info(f"Loaded {total_rows} rows of data +++")
        logger.info(f"Original columns from CSV: {df.columns.tolist()} +++")

        # Store original column names and keep them unchanged
        original_columns = df.columns.tolist()
        logger.info("Using original column names from CSV file +++")

        # Calculate chunk sizes and prepare chunks using numpy
        NUM_PROCESSES = 7
        logger.info("Splitting data into chunks using numpy.array_split +++")
        
        # Use numpy to split the DataFrame into equal chunks
        df_chunks = np.array_split(df, NUM_PROCESSES) # data frame list
        # chunks = [
        #     (i + 1, chunk.copy(), original_columns)
        #     for i, chunk in enumerate(df_chunks)
        # ]
        # no need tocopy

        # required_columns = {"column1", "column2", "column3"}  # Замени на реальные имена колонок
        # missing_columns = required_columns - set(df_chunk.columns)
        # if missing_columns:
        #     logger.error(f"Missing columns in chunk {chunk_id}: {missing_columns}")
        #     raise ValueError(f"Missing columns in chunk {chunk_id}: {missing_columns}")

        chunks = [(i + 1, chunk, original_columns) for i, chunk in enumerate(df_chunks)]

        logger.info(f"CHUNKS PRIOR TO PROCESSING ++++++++++++ {chunks}")

        a = chunks

        for chunk_id, df_chunk, original_columns in chunks:
            # Логируем все колонки в текущем чанке
            logger.info(f"Chunk {chunk_id} columns: {df_chunk.columns.tolist()} @@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        logger.info(f"Created {len(chunks)} chunks +++")
        for i, (chunk_id, chunk_df, _) in enumerate(chunks):
            logger.info(
                f"Chunk {chunk_id}: {len(chunk_df)} rows "
                f"({len(chunk_df)/total_rows*100:.1f}%) +++"
            )

        # Add debug logging for initial DataFrame columns
        logger.info("Debug: Initial DataFrame columns: +++")
        logger.info(f"Original columns from CSV: {original_columns} +++")
        logger.info("Sample of first few rows: +++")
        logger.info(f"{df.head()} +++")

        # Process chunks using multiprocessing pool
        logger.info("Starting parallel processing with Pool... +++")
        with Pool(processes=NUM_PROCESSES) as pool:
            chunk_results = pool.map(process_chunk, chunks)

        logger.info(f"CHUNK_RESULTS ++++++++++++++++ {chunk_results}") # Empty DataFrame
        logger.info("All chunks processed, concatenating results... +++")
        
        # Concatenate all resulting DataFrames
        final_df = pd.concat(chunk_results, ignore_index=True)
        logger.info(f"FINAL_DF_AFTER_CONCATENATION {final_df}") # Empty DataFrame
        final_df['н п.п.'] = range(1, len(final_df) + 1)
        
        # Add debug logging for final DataFrame
        logger.info("Debug: Final DataFrame after processing: +++")
        logger.info(f"Final columns: {final_df.columns.tolist()} +++")
        logger.info("Sample of processed data: +++")
        logger.info(f"{final_df.head()} +++")

        # Add detailed column inspection before brand processing
        logger.info("Debug: Detailed column inspection before brand processing: +++")
        logger.info(f"All available columns: {final_df.columns.tolist()} +++")
        logger.info("Checking each key column individually: +++")
        
        # First print the exact column names from DataFrame for comparison
        logger.info("Exact column names in DataFrame: +++")
        for col in final_df.columns:
            logger.info(f"'{col}' - Length: {len(col)} - Repr: {repr(col)} +++")
        
        # Then check each key column
        key_columns = [
            'Номенклатура Товарів Послуг',
            'Ціна Без ПДВ',
            "Кількість об’єм  Обсяг"  # Using actual column name
        ]
        for col in key_columns:
            try:
                logger.info(f"Checking column '{col}' - Length: {len(col)} - Repr: {repr(col)} +++")
                if col in final_df.columns:
                    logger.info(f"Column '{col}' exists in DataFrame +++")
                    logger.info(f"Sample of '{col}': {final_df[col].head()} +++")
                else:
                    logger.info(f"Column '{col}' NOT FOUND in DataFrame +++")
                    logger.info(f"Available columns: {final_df.columns.tolist()} +++")
            except Exception as e:
                logger.error(f"Error checking column '{col}': {str(e)} +++")
                logger.error(f"Error type: {type(e)} +++")
                logger.error(f"Error repr: {repr(e)} +++")

        # Save products catalog
        final_df.to_csv(
            'products_catalog.csv',
            sep=';',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            float_format='%.2f'
        )
        logger.info("Products catalog saved successfully")
        
        # Create non-lubricant products catalog
        logger.info("Creating non-lubricant products catalog... $$$$")
        logger.info(f"Total products before filtering: {len(final_df)} $$$$")
        logger.info("Sample of products being checked:")
        logger.info(final_df['Номенклатура Товарів Послуг'].head().to_string() + " $$$$")

        logger.info(f"FINAL_DF_CONTENT {final_df}") # Empty DataFrame
        logger.info(f"FINAL_DF_KEYS {final_df['Номенклатура Товарів Послуг']}") #OUTPUT - Series([], Name: Номенклатура Товарів Послуг, dtype: object)

        non_lubricant_mask = final_df['Номенклатура Товарів Послуг'].apply(
            lambda x: not is_lubricant_related(x)
        )

        u = non_lubricant_mask

        # Log the count of non-lubricant products
        logger.info(f"Number of non-lubricant products found: {non_lubricant_mask.sum()} $$$$")
        logger.info(f"Number of lubricant products found: {(~non_lubricant_mask).sum()} $$$$")

        non_lubricant_df = final_df[non_lubricant_mask].copy()

        m = non_lubricant_df

        logger.info(f"Created non_lubricant_df with {len(non_lubricant_df)} rows $$$$%%%%%---%%%") #0 ROWS !!!!

        logger.info(f"NON-LUB DataFrame info: {final_df[~non_lubricant_mask]} $$$$--****---") #EMPTY LIST !!!!!
        if len(non_lubricant_df) == 0:
            logger.warning("Warning: non_lubricant_df is empty! $$$$")
            logger.info("Sample of products that were classified as lubricants:")
            # First log what columns are actually available
            logger.info(f"Available columns in lubricant DataFrame: {final_df[~non_lubricant_mask].columns.tolist()} $$$$")
            logger.info(f"Shape of lubricant DataFrame: {final_df[~non_lubricant_mask].shape} $$$$")
            # Then try to access columns safely
            try:
                logger.info("Attempting to show first 5 rows of lubricant products: $$$$")
                for col in final_df[~non_lubricant_mask].columns:
                    logger.info(f"Column '{col}' sample:\n{final_df[~non_lubricant_mask][col].head().to_string()} $$$$")
            except Exception as e:
                logger.error(f"Error accessing DataFrame columns: {str(e)} $$$$")
            lubricant_sample = final_df[~non_lubricant_mask]['Номенклатура Товарів Послуг'].head()
        #MAKE DEBUGGING HERE !!!!!
        non_lubricant_df['н п.п.'] = range(1, len(non_lubricant_df) + 1)
        
        non_lubricant_df.to_csv(
            'non_lubricant_products.csv',
            sep=';',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            float_format='%.2f'
        )
        logger.info("Non-lubricant products catalog saved successfully")
        
        # Add debug logging before brand processing
        logger.info("Debug: DataFrame before brand processing:")
        logger.info(f"Columns available: {final_df.columns.tolist()}")
        logger.info("Sample of key columns:")
        logger.info(final_df[['Номенклатура Товарів Послуг', 'Ціна Без ПДВ', "Кількість об’єм  Обсяг"]].head())
        
        # Create brands catalog
        logger.info("Creating brands catalog...")
        brands_data = []
        
        for product_name in final_df['Номенклатура Товарів Послуг'].unique():
            brand = extract_brand(product_name)
            brand_rows = final_df[
                final_df['Номенклатура Товарів Послуг'].apply(
                    lambda x: extract_brand(x) == brand
                )
            ]
            # total_quantity = brand_rows["Кількість об’єм  Обсяг"].sum()
            brand_rows["Кількість об’єм  Обсяг"] = pd.to_numeric(brand_rows["Кількість об’єм  Обсяг"], errors="coerce")
            total_quantity = brand_rows["Кількість об’єм  Обсяг"].sum()
            #available keys here
            logger.info(f"some code {brand_rows}")
            # avg_price = brand_rows['Ціна Без ПДВ'].mean()
            # brand_rows['Ціна Без ПДВ'] = pd.to_numeric(brand_rows['Ціна Без ПДВ'], errors='coerce')
            brand_rows.loc[:, 'Ціна Без ПДВ'] = pd.to_numeric(brand_rows['Ціна Без ПДВ'], errors='coerce')
            avg_price = brand_rows['Ціна Без ПДВ'].mean()
            total_quantity = pd.to_numeric(total_quantity, errors='coerce')
            brands_data.append({
                'н п.п.': len(brands_data) + 1,
                'Дата': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'Номенклатура Товарів Послуг': brand,
                'Одиниця Виміру Товару  Послуги': 'шт',
                "Кількість об’єм  Обсяг": total_quantity,
                'Сума Без ПДВ': total_quantity * avg_price,
                'Ціна Без ПДВ': avg_price
            })
        
        brands_df = pd.DataFrame(brands_data)
        brands_df = brands_df.reindex(columns=original_columns)
        
        brands_df.to_csv(
            'brands_catalog.csv',
            sep=';',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            float_format='%.2f'
        )
        logger.info("Brands catalog saved successfully")
        
        # Create summary report
        logger.info("Creating summary report...")
        logger.info(f"some code {non_lubricant_df} **********************")
        summary_data = [
            {
                'н п.п.': 1,
                'Дата': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'Номенклатура Товарів Послуг': f'Total unique brands: {len(brands_df)}',
                'Одиниця Виміру Товару  Послуги': 'шт',
                "Кількість об’єм  Обсяг": len(brands_df),
                'Сума Без ПДВ': brands_df['Ціна Без ПДВ'].sum(),
                'Ціна Без ПДВ': 0
            },
            {
                'н п.п.': 2,
                'Дата': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'Номенклатура Товарів Послуг': f'Total unique products: {len(final_df)}',
                'Одиниця Виміру Товару  Послуги': 'шт',
                "Кількість об’єм  Обсяг": len(final_df),
                'Сума Без ПДВ': final_df['Ціна Без ПДВ'].sum(),
                'Ціна Без ПДВ': 0
            },
            {
                'н п.п.': 3,
                'Дата': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'Номенклатура Товарів Послуг': f'Non-lubricant products: {len(non_lubricant_df)}',
                'Одиниця Виміру Товару  Послуги': 'шт',
                "Кількість об’єм  Обсяг": len(non_lubricant_df),
                'Сума Без ПДВ': non_lubricant_df['Ціна Без ПДВ'].sum(),
                'Ціна Без ПДВ': 0
            }
        ]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.reindex(columns=original_columns)
        
        summary_df.to_csv(
            'processing_summary.csv',
            sep=';',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            float_format='%.2f'
        )
        logger.info("Summary report saved successfully")
        
        logger.info("All processing completed successfully")

    except Exception as e:
        logger.error(
            f"An error occurred during processing: {str(e)}", 
            exc_info=True
        )
        raise


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
