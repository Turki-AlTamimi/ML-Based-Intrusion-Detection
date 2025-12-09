import re
import os
import pandas as pd

def parse_results_file(filename: str, split_label: str) -> pd.DataFrame:
    rows = []
    current_clf = None
    metrics = {}

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Detect classifier block
        if line.startswith("Classifying with "):
            if current_clf and metrics:
                metrics["classifier"] = current_clf
                metrics["split"] = split_label
                rows.append(metrics)
                metrics = {}

            # Extract classifier name
            current_clf = line.replace("Classifying with ", "").split(".")[0].strip()

        elif line.startswith("Average TPR"):
            metrics["avg_tpr"] = float(line.split("=")[1].strip())

        elif line.startswith("Average FPR"):
            metrics["avg_fpr"] = float(line.split("=")[1].strip())

        elif line.startswith("Average Accuracy"):
            metrics["avg_accuracy"] = float(line.split("=")[1].strip())

        elif line.startswith("Average F1 Score"):
            metrics["avg_f1"] = float(line.split("=")[1].strip())

        elif line.startswith("Average Precision"):
            metrics["avg_precision"] = float(line.split("=")[1].strip())

            # block finished
            metrics["classifier"] = current_clf
            metrics["split"] = split_label
            rows.append(metrics)
            metrics = {}

    return pd.DataFrame(rows)


def main():
    # Detect result files automatically
    all_files = os.listdir(".")
    result_files = [f for f in all_files if f.endswith(".results.txt")]

    if not result_files:
        print("❌ No results files found in this folder!")
        return

    all_data = []

    for f in result_files:
        print(f"Processing: {f}")

        # extract split number e.g. data.csv_20.results.txt → 20
        try:
            split_num = re.findall(r"_(\d+)\.results", f)[0]
        except:
            split_num = "N/A"

        df = parse_results_file(f, split_num + "%")
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    # Output CSV
    output_name = "combined_results.csv"
    if os.path.exists(output_name):
        try:
          os.remove(output_name)
        except:
         print("❌ Cannot overwrite combined_results.csv. Make sure it is closed.")
        return
    final_df.to_csv(output_name, index=False)

    print("\n✅ DONE! File saved:", output_name)
    print(final_df)


if __name__ == "__main__":
    main()
