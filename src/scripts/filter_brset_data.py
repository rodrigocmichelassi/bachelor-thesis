import os
import pandas as pd

def main():
    count = 0
    skipped = 0

    csv_path = '/home/rodrigocm/tcc/data/captions.csv'

    brset_path = '/scratch/datasets/retina/brset/imgs'
    dest_path = '/home/rodrigocm/tcc/data/images'

    brset_df = pd.read_csv(csv_path)
    os.makedirs(dest_path, exist_ok=True)

    num_images = brset_df['image_id'].nunique()

    print(f"Linking {num_images} images to {dest_path}")
    for _, row in brset_df.iterrows():
        image_id = row['image_id']

        source_file = os.path.join(brset_path, image_id)
        dest_file = os.path.join(dest_path, image_id)

        if os.path.exists(dest_file) or os.path.islink(dest_file):
            skipped += 1
            continue

        if os.path.exists(source_file):
            os.symlink(source_file, dest_file)
            count += 1
        else:
            print(f"Warning: File not found: {source_file}")

    print(f"Successfully linked {count} images ({skipped} already existed) to {dest_path}!")

if __name__ == '__main__':
    main()