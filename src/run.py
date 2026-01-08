import argparse

from core.utils import download_data, clean_data


def main():

    parser = argparse.ArgumentParser(description="Screen GC-ADDA4TB surrogate models on a specified Enamine 10B data chunk.")
    parser.add_argument("--chunk_name", type=str, required=True, help="Name of the data chunk to process.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output CSV file.")
    args = parser.parse_args()

    chunk_name = args.chunk_name
    output_dir = args.output_dir
    

    download_data(output_dir, chunk_name)
    
    # something here
    
    # clean_data(output_dir, chunk_name)


if __name__ == "__main__":
    main()