import pandas as pd
import glob


def find_pair(base_path, base_suffix, pair_suffix):
    pair_path = base_path.replace(base_suffix, pair_suffix)
    base_csv = pd.read_csv(base_path)
    pair_csv = pd.read_csv(pair_path)
    df_merged = pd.concat([base_csv, pair_csv], ignore_index=True)
    return df_merged


if __name__ == '__main__':
    pair_suffix = "_general100s"
    base_suffix = "_general_100"
    output_suffix = "_general200"
    all_bases = glob.glob(f"./*{base_suffix}.csv")
    for b in all_bases:
        print(b)
        result = find_pair(b, base_suffix=base_suffix, pair_suffix=pair_suffix)
        assert len(result.drop_duplicates()) == len(result)
        result.to_csv(b.replace(base_suffix, output_suffix))
