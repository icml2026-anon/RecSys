"""
Convert Amazon Beauty dataset to RecBole format
This script downloads and converts the Amazon Beauty 2014 dataset
"""
import json
import gzip
import pandas as pd
import os
from datetime import datetime

def download_amazon_beauty():
    """Download Amazon Beauty dataset from available mirrors"""
    import urllib.request
    import ssl

    # Disable SSL verification for problematic certificates
    ssl_context = ssl._create_unverified_context()

    # Try multiple sources
    urls = [
        "https://jmcauley.ucsd.edu/data/amazon/categoryFilesSmall/reviews_Beauty_5.json.gz",
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
    ]

    output_file = "reviews_Beauty_5.json.gz"

    for url in urls:
        try:
            print(f"Trying to download from: {url}")

            if url.startswith('https'):
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
                urllib.request.install_opener(opener)

            urllib.request.urlretrieve(url, output_file)

            # Check if it's a valid gzip file
            with gzip.open(output_file, 'rt') as f:
                f.readline()  # Try to read first line

            print(f"✓ Successfully downloaded from {url}")
            return output_file
        except Exception as e:
            print(f"✗ Failed: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            continue

    raise Exception("Could not download from any source")

def convert_to_recbole_format(input_file, output_dir="Amazon-Beauty"):
    """Convert Amazon review JSON to RecBole format"""

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading {input_file}...")
    reviews = []

    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line.strip())
            reviews.append({
                'user_id': review.get('reviewerID', ''),
                'item_id': review.get('asin', ''),
                'rating': float(review.get('overall', 0)),
                'timestamp': int(review.get('unixReviewTime', 0))
            })

    df = pd.DataFrame(reviews)

    print(f"Total reviews: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique items: {df['item_id'].nunique()}")

    # Save in RecBole format
    inter_file = os.path.join(output_dir, "Amazon-Beauty.inter")
    df.to_csv(inter_file, sep='\t', index=False)

    print(f"✓ Saved to {inter_file}")

    # Create README
    readme_content = f"""# Amazon Beauty Dataset

Converted from Amazon Product Reviews (2014)
Source: Julian McAuley's Amazon review dataset

## Statistics
- Total interactions: {len(df)}
- Unique users: {df['user_id'].nunique()}
- Unique items: {df['item_id'].nunique()}
- Rating range: {df['rating'].min()} - {df['rating'].max()}

## Files
- Amazon-Beauty.inter: User-item interaction file

## Format
Tab-separated values with columns:
- user_id: User identifier
- item_id: Item identifier (ASIN)
- rating: Rating score (1-5)
- timestamp: Unix timestamp

Converted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write(readme_content)

    return output_dir

if __name__ == '__main__':
    try:
        # Check if file already exists
        input_file = "reviews_Beauty_5.json.gz"
        if not os.path.exists(input_file):
            # Download dataset
            input_file = download_amazon_beauty()
        else:
            print(f"Using existing file: {input_file}")

        # Convert to RecBole format
        output_dir = convert_to_recbole_format(input_file)

        print(f"\n{'='*60}")
        print(f"✓ Amazon Beauty dataset ready at: {output_dir}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease manually download the dataset from:")
        print("  https://jmcauley.ucsd.edu/data/amazon/")
        print("And place 'reviews_Beauty_5.json.gz' in this directory")
