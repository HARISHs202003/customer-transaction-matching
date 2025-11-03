import pandas as pd
from rapidfuzz import fuzz, process

# ---- Step 1: Load Excel File ----
file_path = "Hiring - Assignment1.xlsx"  # change if needed

customers = pd.read_excel(file_path, sheet_name="Customers")
transactions = pd.read_excel(file_path, sheet_name="Transations")

# ---- Step 2: Prepare / Clean Text ----
def clean(text):
    if pd.isna(text):
        return ""
    return str(text).lower().replace(" ", "").replace("-", "").replace("_", "").replace("/", "")

customers["clean_name"] = customers["Invoice Customer Name"].apply(clean)
customers["clean_alias"] = customers["Bank Alias Name"].apply(clean)
customers["clean_upi"] = customers["UPI Handle"].apply(clean)
customers["clean_phone"] = customers["Invoice Phone"].apply(lambda x: str(x) if not pd.isna(x) else "")

transactions["clean_desc"] = transactions["Description"].apply(clean)
transactions["clean_ref"] = transactions["Ref No./Cheque No."].apply(clean)

# ---- Step 3: Matching Function ----
GENERIC_WORDS = {"transfer","towards","payment","invoice","credit","receipt","from","by","inb","imps","neft","upi"}

def is_valid_alias(alias):
    alias = alias.lower()
    if len(alias) < 6:
        return False
    return not any(word in alias for word in GENERIC_WORDS)

def match_transaction(desc, ref):
    best_score = 0
    best_customer = None
    
    for _, cust in customers.iterrows():
        score = 0
        
        # Strongest: UPI Handle match
        if cust["clean_upi"] and len(cust["clean_upi"]) >= 6:
            if cust["clean_upi"] in desc or cust["clean_upi"] in ref:
                score = 100
        
        # Alias match (only if alias is meaningful)
        if is_valid_alias(cust["clean_alias"]):
            if cust["clean_alias"] in desc or cust["clean_alias"] in ref:
                score = max(score, 90)
        
        # Phone match (must be exact)
        if cust["clean_phone"] and cust["clean_phone"] in desc:
            score = max(score, 75)
        
        # Fuzzy name match (only count if similar enough)
        fuzzy_score = fuzz.partial_ratio(cust["clean_name"], desc)
        if fuzzy_score >= 75:
            score = max(score, fuzzy_score)
        
        # Keep highest score
        if score > best_score:
            best_score = score
            best_customer = cust
    
    return best_customer, best_score

# ---- Step 4: Apply Matching ----
customer_ids = []
customer_names = []
customer_phones = []
match_scores = []

for _, row in transactions.iterrows():
    cust, score = match_transaction(row["clean_desc"], row["clean_ref"])
    
    if cust is not None and score >= 70:  # minimum confidence threshold
        customer_ids.append(cust["Customer Id"])
        customer_names.append(cust["Invoice Customer Name"])
        customer_phones.append(cust["Invoice Phone"])
        match_scores.append(score)
    else:
        customer_ids.append("")
        customer_names.append("")
        customer_phones.append("")
        match_scores.append(0)

transactions["Matched Customer Id"] = customer_ids
transactions["Matched Name"] = customer_names
transactions["Matched Phone"] = customer_phones
transactions["Match Score"] = match_scores

# ---- Step 5: Save Output ----
output_file = "Matched_Transactions.xlsx"
transactions.to_excel(output_file, index=False)

print("✅ Matching Completed")
print(f"📁 Results saved to: {output_file}")

# ---- Step 6: Match Statistics ----
matched = sum(transactions["Match Score"] > 0)
unmatched = len(transactions) - matched

print("\n📊 Matching Summary:")
print(f"Total Transactions: {len(transactions)}")
print(f"Matched: {matched}")
print(f"Unmatched: {unmatched}")

print("\nMatch Score Distribution:")
print(transactions["Match Score"].value_counts(bins=[0,40,60,80,90,100]))
import matplotlib.pyplot as plt

# Define bins and labels
bins = [0, 40, 60, 80, 90, 100]
labels = ['0-40', '40-60', '60-80', '80-90', '90-100']

# Create bucketed groups
bucketed = pd.cut(transactions["Match Score"], bins=bins, labels=labels, include_lowest=True)

# Count values in each bucket
bucket_counts = bucketed.value_counts().sort_index()

# Plot
plt.figure(figsize=(8,5))
plt.bar(bucket_counts.index, bucket_counts.values)
plt.xlabel("Match Score Range")
plt.ylabel("Number of Transactions")
plt.title("Match Score Confidence Buckets")
plt.show()


