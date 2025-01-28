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
        logger.info(f"Processing chunk {chunk_id}")
        
        # Extract information
        product_col = 'Номенклатура Товарів Послуг'
        df_chunk['brand'] = df_chunk[product_col].apply(extract_brand)
        df_chunk['volume_liters'] = df_chunk[product_col].apply(extract_volume)
        df_chunk['is_lubricant'] = df_chunk[product_col].apply(
            is_lubricant_related
        )
        
        # Create products catalog for this chunk
        products_catalog = (
            df_chunk.groupby(product_col)
            .agg({
                'brand': 'first',
                'unit': 'first',
                'volume_liters': 'first',
                'is_lubricant': 'first'
            })
            .reset_index(name='product_name')  # Rename the index column
        )
        
        products_catalog['volume_liters'] = (
            products_catalog['volume_liters'].fillna(0)
        )
        
        # Create result DataFrame
        products_df = pd.DataFrame(columns=original_columns)
        
        for idx, row in enumerate(products_catalog.itertuples(), 1):
            products_df.loc[idx] = [
                idx,
                pd.Timestamp.now().strftime('%Y-%m-%d'),
                row.product_name,  # Use the renamed column
                row.unit,
                row.volume_liters,
                0,
                0
            ]
        
        logger.info(f"Completed processing chunk {chunk_id}")
        return products_df
        
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
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
            'масло', 'смазка', 'жидкость', 'oil', 'fluid', 'grease',
            'лубрикант', 'очиститель', 'cleaner', 'присадка', 'additive'
        ]
        return any(
            keyword.lower() in str(product_name).lower() 
            for keyword in lubricant_keywords
        )
    except Exception as e:
        logger.error(
            f"Error checking lubricant relation for '{product_name}': {str(e)}"
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
        logger.info("Starting data processing...")
        
        # Read the CSV file
        logger.info("Reading input CSV file...")
        try:
            df = pd.read_csv('output.csv', sep=';', encoding='utf-8')
            logger.info("Successfully read CSV with UTF-8 encoding")
        except UnicodeDecodeError:
            logger.info("UTF-8 encoding failed, trying CP1251...")
            df = pd.read_csv('output.csv', sep=';', encoding='cp1251')
            logger.info("Successfully read CSV with CP1251 encoding")

        print("INITIAL_DF***********", df.columns)
        
        # Log initial data info
        total_rows = len(df)
        logger.info(f"Loaded {total_rows} rows of data")
        logger.info(f"Original columns: {df.columns.tolist()}")

        # Clean column names and get original column names for reference
        original_columns = df.columns.tolist()
        df.columns = [
            'id', 'date', 'Номенклатура Товарів Послуг', 'unit', 'quantity',
            'total_without_vat', 'price_without_vat'
        ]
        logger.info("Column names standardized")
        print("STANDARDIZED_DF***********", df.columns)

        # Calculate chunk sizes and prepare chunks using numpy
        NUM_PROCESSES = 7
        logger.info("Splitting data into chunks using numpy.array_split")
        
        # Use numpy to split the DataFrame into equal chunks
        df_chunks = np.array_split(df, NUM_PROCESSES)
        chunks = [
            (i + 1, chunk.copy(), original_columns) 
            for i, chunk in enumerate(df_chunks)
        ]
        
        logger.info(f"Created {len(chunks)} chunks")
        for i, (chunk_id, chunk_df, _) in enumerate(chunks):
            logger.info(
                f"Chunk {chunk_id}: {len(chunk_df)} rows "
                f"({len(chunk_df)/total_rows*100:.1f}%)"
            )

        # Process chunks using multiprocessing pool
        logger.info("Starting parallel processing with Pool...")
        with Pool(processes=NUM_PROCESSES) as pool:
            # Map process_chunk function to chunks and get DataFrames directly
            chunk_results = pool.map(process_chunk, chunks)
        
        logger.info("All chunks processed, concatenating results...")
        
        # Concatenate all resulting DataFrames
        final_df = pd.concat(chunk_results, ignore_index=True, axis=0)
        
        # Reindex to ensure sequential numbering
        final_df['id'] = range(1, len(final_df) + 1)
        print("FINAL_DF***********", final_df.columns)
        
        # Save final result
        final_df.to_csv(
            'products_catalog.csv',
            sep=';',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            float_format='%.2f'
        )
        
        logger.info("Final products catalog saved successfully")
        
        # Create non-lubricant products catalog
        logger.info("Creating non-lubricant products catalog...")
        non_lubricant_mask = ~final_df['Номенклатура Товарів Послуг'].apply(
            is_lubricant_related
        )
        non_lubricant_df = final_df[non_lubricant_mask].copy()
        non_lubricant_df['id'] = range(1, len(non_lubricant_df) + 1)
        print("NON_LUBRICANT_DF***********", non_lubricant_df.columns)
        
        non_lubricant_df.to_csv(
            'non_lubricant_products.csv',
            sep=';',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            float_format='%.2f'
        )
        
        # Create brands catalog
        logger.info("Creating brands catalog...")
        brands_df = pd.DataFrame(columns=original_columns)
        brand_counts = final_df.groupby('Номенклатура Товарів Послуг').apply(
            lambda x: x['brand'].iloc[0]
        ).value_counts()
        
        for idx, (brand, count) in enumerate(brand_counts.items(), 1):
            brands_df.loc[idx] = [
                idx,
                pd.Timestamp.now().strftime('%Y-%m-%d'),
                brand,
                'шт',
                count,
                0,
                0
            ]
        
        print("BRANDS_DF***********", brands_df.columns)
        brands_df.to_csv(
            'brands_catalog.csv',
            sep=';',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            float_format='%.2f'
        )
        
        # Create summary report
        logger.info("Creating summary report...")
        summary_df = pd.DataFrame(columns=original_columns)
        summary_df.loc[1] = [
            1,
            pd.Timestamp.now().strftime('%Y-%m-%d'),
            f'Total unique brands: {len(brands_df)}',
            'шт',
            len(brands_df),
            0,
            0
        ]
        summary_df.loc[2] = [
            2,
            pd.Timestamp.now().strftime('%Y-%m-%d'),
            f'Total unique products: {len(final_df)}',
            'шт',
            len(final_df),
            0,
            0
        ]
        summary_df.loc[3] = [
            3,
            pd.Timestamp.now().strftime('%Y-%m-%d'),
            f'Non-lubricant products: {len(non_lubricant_df)}',
            'шт',
            len(non_lubricant_df),
            0,
            0
        ]
        
        print("SUMMARY_DF***********", summary_df.columns)
        summary_df.to_csv(
            'processing_summary.csv',
            sep=';',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL,
            float_format='%.2f'
        )
        
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