import re
import os
import pandas as pd
import matplotlib.pyplot as plt 
OUTPUT_DIR = "plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ---------- 1) function to parse one results file ----------

def parse_results_file(filename: str, split_label: str) -> pd.DataFrame:
    """
    Reads a *_results.txt file and extracts:
    classifier name + Average TPR/FPR/Accuracy/F1/Precision
    """
    rows = []
    current_clf = None
    metrics = {}

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Detect classifier block
        if line.startswith("Classifying with "):
            # لو فيه مصنف سابق وله بيانات، نحفظه
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

# ---------- 2) main: parse 20% & 50% and plot ----------

def main():
    # لو تبغى ملفات ثانية غيّر الأسماء هنا
    file_20 = "data.csv_20.results.txt"
    file_50 = "data.csv_50.results.txt"

    if not os.path.exists(file_20) or not os.path.exists(file_50):
        print("❌ Make sure data.csv_20.results.txt and data.csv_50.results.txt exist in this folder.")
        return

    df20 = parse_results_file(file_20, "20%")
    df50 = parse_results_file(file_50, "50%")

    # دمج الاثنين في جدول واحد
    df = pd.concat([df20, df50], ignore_index=True)

    print("Parsed results:")
    print(df)

    classifiers = df["classifier"].unique()
    x = range(len(classifiers))
    width = 0.35
    x1 = [i - width/2 for i in x]
    x2 = [i + width/2 for i in x]

    # =============== 1) TPR ===============
    plt.figure(figsize=(10, 6))
    tpr_20 = [df[(df["classifier"] == c) & (df["split"] == "20%")]["avg_tpr"].values[0] for c in classifiers]
    tpr_50 = [df[(df["classifier"] == c) & (df["split"] == "50%")]["avg_tpr"].values[0] for c in classifiers]

    plt.bar(x1, tpr_20, width=width, label="20% test")
    plt.bar(x2, tpr_50, width=width, label="50% test")

    plt.xticks(list(x), classifiers, rotation=30, ha="right")
    plt.ylabel("Average TPR")
    plt.title("Comparison of Average TPR (20% vs 50%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "compare_tpr_20_50.png"), dpi=300)

    # =============== 2) FPR ===============
    plt.figure(figsize=(10, 6))
    fpr_20 = [df[(df["classifier"] == c) & (df["split"] == "20%")]["avg_fpr"].values[0] for c in classifiers]
    fpr_50 = [df[(df["classifier"] == c) & (df["split"] == "50%")]["avg_fpr"].values[0] for c in classifiers]

    plt.bar(x1, fpr_20, width=width, label="20% test")
    plt.bar(x2, fpr_50, width=width, label="50% test")

    plt.xticks(list(x), classifiers, rotation=30, ha="right")
    plt.ylabel("Average FPR")
    plt.title("Comparison of Average FPR (20% vs 50%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"compare_fpr_20_50.png"), dpi=300)

    # =============== 3) Accuracy ===============
    plt.figure(figsize=(10, 6))
    acc_20 = [df[(df["classifier"] == c) & (df["split"] == "20%")]["avg_accuracy"].values[0] for c in classifiers]
    acc_50 = [df[(df["classifier"] == c) & (df["split"] == "50%")]["avg_accuracy"].values[0] for c in classifiers]

    plt.bar(x1, acc_20, width=width, label="20% test")
    plt.bar(x2, acc_50, width=width, label="50% test")

    plt.xticks(list(x), classifiers, rotation=30, ha="right")
    plt.ylabel("Average Accuracy")
    plt.title("Comparison of Average Accuracy (20% vs 50%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"compare_accuracy_20_50.png"), dpi=300)

    # =============== 4) F1 Score ===============
    plt.figure(figsize=(10, 6))
    f1_20 = [df[(df["classifier"] == c) & (df["split"] == "20%")]["avg_f1"].values[0] for c in classifiers]
    f1_50 = [df[(df["classifier"] == c) & (df["split"] == "50%")]["avg_f1"].values[0] for c in classifiers]

    plt.bar(x1, f1_20, width=width, label="20% test")
    plt.bar(x2, f1_50, width=width, label="50% test")

    plt.xticks(list(x), classifiers, rotation=30, ha="right")
    plt.ylabel("Average F1 Score")
    plt.title("Comparison of Average F1 Score (20% vs 50%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"compare_f1_20_50.png"), dpi=300)

    print("\n✅ Saved plots:")
    print("  - compare_tpr_20_50.png")
    print("  - compare_fpr_20_50.png")
    print("  - compare_accuracy_20_50.png")
    print("  - compare_f1_20_50.png")


if __name__ == "__main__":
    main()
