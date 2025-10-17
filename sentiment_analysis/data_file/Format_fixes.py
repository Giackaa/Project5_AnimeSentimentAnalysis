import csv
import pandas as pd

def fix_csv(input_path, output_path):
    fixed_rows = []
    buffer = ""

    # Read header first
    with open(input_path, "r", encoding="utf-8") as infile:
        header = infile.readline().strip()
        fixed_rows.append(header.split(","))  # split header into columns

        for line in infile:
            buffer += line.strip("\n") + " "

            # If quotes are balanced, parse and flush buffer
            if buffer.count('"') % 2 == 0:
                try:
                    row = next(csv.reader([buffer]))
                    fixed_rows.append(row)
                except Exception as e:
                    print(f"⚠️ Skipping broken line: {buffer[:80]}... ({e})")
                buffer = ""

    # Write cleaned file
    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(fixed_rows)

    print(f"✅ Fixed file written to: {output_path}")

def PutQuotation (file):
    # Read the CSV file
    df = pd.read_csv(file)

    # Add quotation marks to each entry in the 'quoted_text' column
    # Only if the value is not empty (or NaN)
    df['quoted_text'] = df['quoted_text'].apply(
        lambda x: f'"{x}"' if pd.notna(x) and x != '' else x
    )

    # Save the modified DataFrame back to CSV
    df.to_csv(file, index=False)

    print("Quotation marks added successfully!")

def putEpRating():
    df = pd.read_csv(r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\TF-IDF\sentiments_repairedCSV.csv")
    ratings = [4.0, 3.2, 3.5, 3.5, 3.9, 3.9, 4.1, 3.9, 4.0, 4.1, 4.2, 4.2, 4.4, 4.6]

    # Create mapping: Ep1 → 4.0, Ep2 → 3.2, ...
    episode_to_rating = {f"Ep{num}": rating for num, rating in enumerate(ratings, start=1)}
    df["ep_rating"] = df["episode"].map(episode_to_rating)
    return df.to_csv(r"C:\Users\jackl\OneDrive\Desktop\Project 5 - Anime Sentiments\TF-IDF\sentiments_repairedCSV.csv", index=False)

putEpRating()


