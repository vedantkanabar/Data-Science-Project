import pickle

# USE THIS FILE TO GENERATE THE TXT FILE FOR ENCODER MAPPING
with open("encoderList.pkl", "rb") as file:
    encoders = pickle.load(file)

output_path = "encoder_mappings.txt"

with open(output_path, "w") as f:
    for column, encoder in encoders.items():
        f.write(f"Column: {column}\n")
        f.write("-" * (8 + len(column)) + "\n")
        
        for i, cls in enumerate(encoder.classes_):
            f.write(f"{cls} -> {i}\n")
        
        f.write("\n") 