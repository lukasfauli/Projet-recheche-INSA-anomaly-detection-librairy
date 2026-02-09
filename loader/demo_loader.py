from loader import load_data
data_dict, extracted_dir = load_data(
    # common CSV options if needed:
    # sep=";", encoding="utf-8", decimal=","
)
print(len(data_dict), "CSV files loaded")
first_key = next(iter(data_dict))
print("Example file:", first_key)
print(data_dict[first_key].head())
