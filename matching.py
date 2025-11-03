import argparse
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    # fuzzywuzzy is requested; python-Levenshtein speeds it up if installed
    from fuzzywuzzy import fuzz, process as fuzz_process
except Exception:  # pragma: no cover
    fuzz = None
    fuzz_process = None


UPI_PATTERN = re.compile(r"\b[\w\.-]+@[a-zA-Z]+\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[- ]?)?(\d{10})(?!\d)")


@dataclass
class MatchResult:
    customer_name: Optional[str]
    customer_id: Optional[str]
    score: int


def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = str(value).strip().lower()
    # Collapse whitespace and remove common punctuation that harms matching as words
    value = re.sub(r"\s+", " ", value)
    return value


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from string cells, keep NaNs
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # Convert empty strings to NA
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
    return df


def extract_upis(text: str) -> List[str]:
    if not text:
        return []
    candidates = [m.group(0).lower() for m in UPI_PATTERN.finditer(text)]
    # sanitize duplicates
    unique = list(dict.fromkeys(candidates))
    return unique


def extract_phones(text: str) -> List[str]:
    if not text:
        return []
    phones = [m.group(1) for m in PHONE_PATTERN.finditer(text)]
    unique = list(dict.fromkeys(phones))
    return unique


def build_customer_indices(customers_df: pd.DataFrame) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Returns:
      upi_to_customer: upi -> (customer_id, customer_name)
      phone_to_customer: phone10 -> (customer_id, customer_name)
      alias_to_customer: alias/name token -> (customer_id, customer_name)
      fuzzy_items: list of (key_text, customer_id, customer_name)
    """
    upi_to_customer: Dict[str, Tuple[str, str]] = {}
    phone_to_customer: Dict[str, Tuple[str, str]] = {}
    alias_to_customer: Dict[str, Tuple[str, str]] = {}
    fuzzy_items: List[Tuple[str, str, str]] = []

    # Column name candidates
    col_id = None
    for candidate in ["Customer ID", "CustomerId", "ID", "Customer_Id"]:
        if candidate in customers_df.columns:
            col_id = candidate
            break
    if not col_id:
        raise ValueError("Customers sheet must contain a 'Customer ID' column")

    name_cols = [c for c in customers_df.columns if c.lower() in ["customer/invoice name", "customer name", "name", "invoice name"]]
    if not name_cols:
        raise ValueError("Customers sheet must contain a customer name column (e.g., 'Customer/Invoice Name')")
    col_name = name_cols[0]

    phone_cols = [c for c in customers_df.columns if c.lower() in ["phone number", "phone", "mobile", "mobile number", "contact"]]
    upi_cols = [c for c in customers_df.columns if c.lower() in ["upi handle", "upi", "upi id", "upiid"]]
    alias_cols = [c for c in customers_df.columns if c.lower() in ["bank alias or account name", "bank alias", "account name", "alias", "bank name"]]

    for _, row in customers_df.iterrows():
        customer_id = str(row[col_id]).strip()
        customer_name_raw = row.get(col_name, "")
        customer_name = normalize_text(customer_name_raw)

        if upi_cols:
            upi_raw = str(row.get(upi_cols[0], "") or "").strip().lower()
            if upi_raw:
                upi_to_customer[upi_raw] = (customer_id, customer_name_raw)

        if phone_cols:
            phone_raw = str(row.get(phone_cols[0], "") or "")
            # Extract any 10-digit phone within stored text
            for phone in extract_phones(phone_raw):
                phone_to_customer[phone] = (customer_id, customer_name_raw)

        # Alias index for exact text matching (as tokens)
        alias_values: List[str] = []
        alias_values.append(customer_name)
        if alias_cols:
            alias_raw = normalize_text(row.get(alias_cols[0], ""))
            if alias_raw:
                alias_values.append(alias_raw)

        for alias_text in alias_values:
            if alias_text:
                alias_to_customer[alias_text] = (customer_id, customer_name_raw)
                fuzzy_items.append((alias_text, customer_id, customer_name_raw))

    return upi_to_customer, phone_to_customer, alias_to_customer, fuzzy_items


def find_exact_name_in_text(text_norm: str, alias_to_customer: Dict[str, Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    # Try exact contains for each alias/name as whole word sequence
    for alias_text, (cid, cname) in alias_to_customer.items():
        if not alias_text:
            continue
        # word boundary match
        pattern = r"(?<!\w)" + re.escape(alias_text) + r"(?!\w)"
        if re.search(pattern, text_norm):
            return cid, cname
    return None


def fuzzy_best(text_norm: str, fuzzy_items: List[Tuple[str, str, str]]) -> Optional[Tuple[str, str, int]]:
    if not fuzz_process or not fuzz:
        return None
    choices = [item[0] for item in fuzzy_items]
    if not choices:
        return None
    best = fuzz_process.extractOne(text_norm, choices, scorer=fuzz.token_set_ratio)
    if not best:
        return None
    best_text, score = best[0], best[1]
    # Map back to cid, cname
    for text, cid, cname in fuzzy_items:
        if text == best_text:
            return cid, cname, score
    return None


def match_transaction_row(description: str,
                          upi_to_customer: Dict[str, Tuple[str, str]],
                          phone_to_customer: Dict[str, Tuple[str, str]],
                          alias_to_customer: Dict[str, Tuple[str, str]],
                          fuzzy_items: List[Tuple[str, str, str]]) -> MatchResult:
    description = description or ""
    description_norm = normalize_text(description)

    # 1) UPI handle match => 100
    for upi in extract_upis(description_norm):
        if upi in upi_to_customer:
            customer_id, customer_name = upi_to_customer[upi]
            return MatchResult(customer_name=customer_name, customer_id=customer_id, score=100)

    # 2) Phone match => 90
    for phone in extract_phones(description):
        if phone in phone_to_customer:
            customer_id, customer_name = phone_to_customer[phone]
            return MatchResult(customer_name=customer_name, customer_id=customer_id, score=90)

    # 3) Exact name/alias substring (word bounded) => 75
    exact = find_exact_name_in_text(description_norm, alias_to_customer)
    if exact:
        cid, cname = exact
        return MatchResult(customer_name=cname, customer_id=cid, score=75)

    # 4) Fuzzy name similarity >= 85 => 60
    fuzzy_hit = fuzzy_best(description_norm, fuzzy_items)
    if fuzzy_hit:
        cid, cname, similarity = fuzzy_hit
        if similarity >= 85:
            return MatchResult(customer_name=cname, customer_id=cid, score=60)

    # 5) No match
    return MatchResult(customer_name=None, customer_id=None, score=0)


def process_workbook(input_path: str, output_path: str, chart_path: Optional[str] = None) -> Dict[str, int]:
    xl = pd.ExcelFile(input_path)
    # Infer sheets robustly
    customers_sheet = None
    transactions_sheet = None
    for sheet in xl.sheet_names:
        low = sheet.strip().lower()
        if customers_sheet is None and ("customer" in low or "master" in low):
            customers_sheet = sheet
        if transactions_sheet is None and ("transaction" in low or "txn" in low):
            transactions_sheet = sheet
    if not customers_sheet or not transactions_sheet:
        # fallback to first two sheets
        if len(xl.sheet_names) >= 2:
            customers_sheet = customers_sheet or xl.sheet_names[0]
            transactions_sheet = transactions_sheet or xl.sheet_names[1]
        else:
            raise ValueError("Workbook must have at least two sheets: Customers and Transactions")

    customers_df = clean_dataframe(xl.parse(customers_sheet))
    transactions_df = clean_dataframe(xl.parse(transactions_sheet))

    upi_to_customer, phone_to_customer, alias_to_customer, fuzzy_items = build_customer_indices(customers_df)

    # Ensure required output columns exist
    output_cols = ["Matched Customer Name", "Matched Customer ID", "Match Score"]
    for col in output_cols:
        if col not in transactions_df.columns:
            transactions_df[col] = None

    description_col = None
    for candidate in ["Description", "description", "Narration", "Details", "Remark", "remarks"]:
        if candidate in transactions_df.columns:
            description_col = candidate
            break
    if not description_col:
        raise ValueError("Transactions sheet must contain a 'Description' column")

    # Process rows
    matched_count = 0
    score_counter: Counter = Counter()

    for idx, row in transactions_df.iterrows():
        description = row.get(description_col, "")
        result = match_transaction_row(
            description=description,
            upi_to_customer=upi_to_customer,
            phone_to_customer=phone_to_customer,
            alias_to_customer=alias_to_customer,
            fuzzy_items=fuzzy_items,
        )
        transactions_df.at[idx, "Matched Customer Name"] = result.customer_name
        transactions_df.at[idx, "Matched Customer ID"] = result.customer_id
        transactions_df.at[idx, "Match Score"] = result.score
        score_counter[result.score] += 1
        if result.score > 0:
            matched_count += 1

    # Write output
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        transactions_df.to_excel(writer, sheet_name=transactions_sheet, index=False)

    # Print summary
    total_txn = len(transactions_df)
    unmatched_count = total_txn - matched_count

    print("Summary:")
    print(f"- Total Transactions: {total_txn}")
    print(f"- Number Matched: {matched_count}")
    print(f"- Number Unmatched: {unmatched_count}")
    print("- Score Distribution:")
    for score in sorted(score_counter.keys(), reverse=True):
        count = score_counter[score]
        bar = "#" * max(1, count)
        print(f"  {score:>3}: {count:>4} {bar}")

    # Optional chart
    if chart_path:
        try:
            import matplotlib.pyplot as plt

            scores_sorted = sorted(score_counter.items())
            x = [s for s, _ in scores_sorted]
            y = [c for _, c in scores_sorted]
            plt.figure(figsize=(6, 4))
            plt.bar([str(v) for v in x], y, color="#4C78A8")
            plt.title("Match Score Distribution")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(chart_path)
            print(f"Saved chart to {chart_path}")
        except Exception as e:  # pragma: no cover
            print(f"Chart generation failed: {e}")

    return {
        "total": total_txn,
        "matched": matched_count,
        "unmatched": unmatched_count,
        "scores": dict(score_counter),
    }


def create_sample_excel(path: str) -> None:
    customers = pd.DataFrame([
        {"Customer ID": "C001", "Customer/Invoice Name": "Ravi Kumar", "Phone Number": "9876543210", "Bank Alias or Account Name": "Ravi K", "UPI Handle": "ravi.k@okaxis"},
        {"Customer ID": "C002", "Customer/Invoice Name": "Meena Traders", "Phone Number": "", "Bank Alias or Account Name": "Meena Tr", "UPI Handle": "meenatraders@ybl"},
        {"Customer ID": "C003", "Customer/Invoice Name": "Arun Singh", "Phone Number": "+91 9123456789", "Bank Alias or Account Name": "Arun S", "UPI Handle": ""},
    ])
    transactions = pd.DataFrame([
        {"Transaction Date": "2025-11-01", "Description": "UPI/Cr/Ref 1234 ravi.k@okaxis Payment received", "Reference Number / UTR": "UTR001", "Amount": 1000, "Branch": "Main"},
        {"Transaction Date": "2025-11-01", "Description": "NEFT/Meena Traders Inv 55", "Reference Number / UTR": "UTR002", "Amount": 2500, "Branch": "Main"},
        {"Transaction Date": "2025-11-02", "Description": "IMPS/From 9123456789/Order 77", "Reference Number / UTR": "UTR003", "Amount": 1800, "Branch": "City"},
        {"Transaction Date": "2025-11-03", "Description": "Transfer from Aroon Sing payment", "Reference Number / UTR": "UTR004", "Amount": 1200, "Branch": "City"},
        {"Transaction Date": "2025-11-03", "Description": "Misc credit", "Reference Number / UTR": "UTR005", "Amount": 500, "Branch": "City"},
    ])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        customers.to_excel(writer, sheet_name="Customers", index=False)
        transactions.to_excel(writer, sheet_name="Transactions", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer–Transaction Matching Automation")
    parser.add_argument("--input", "-i", default="input.xlsx", help="Path to input Excel file")
    parser.add_argument("--output", "-o", default="Matched_Transactions.xlsx", help="Path to output Excel file")
    parser.add_argument("--chart", default=None, help="Optional path to save score distribution chart (PNG)")
    parser.add_argument("--make-sample", action="store_true", help="Create a sample input.xlsx and run matching on it")
    args = parser.parse_args()

    if args.make_sample:
        sample_path = args.input
        print(f"Creating sample workbook at {sample_path}...")
        create_sample_excel(sample_path)

    summary = process_workbook(args.input, args.output, chart_path=args.chart)
    # Enable non-interactive repeated runs by exiting 0


if __name__ == "__main__":
    main()

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
