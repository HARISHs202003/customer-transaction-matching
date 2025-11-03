customer-transaction-matching

# Customer–Transaction Matching System

# Customer–Transaction Matching System

This project automates the process of matching bank transaction records with customer master data. The goal is to accurately identify which customer made each payment based on UPI IDs, bank aliases, name patterns, and fuzzy matching. The program also calculates a match confidence score to help validate results.

---

## Project Structure

customer-transaction-matching/
│
├── matching.py                # Main matching script
├── Matched_Transactions.xlsx  # Output file containing match results & confidence scores
├── Hiring - Assignment1.xlsx  # Input dataset (Customer + Transactions data)
└── requirements.txt           # Python dependency list

---

## Objective

Bank statement narration rarely contains the exact customer name. This system resolves that by intelligently matching transaction references with customer identities using multiple matching techniques and scoring logic.

---

## Matching Logic Breakdown

1. **UPI Handle Matching** – Direct match, highest confidence.
2. **Alias / Nickname / Invoice Name Search** – Searches transaction text for known customer references.
3. **Fuzzy Text Similarity (RapidFuzz)** – Measures name similarity.
4. **Weighted Scoring System** – Multiple indicators combined into a final confidence score.

| Score Range | Interpretation |
|------------|----------------|
| 90–100      | ✅ High confidence: Safe match |
| 70–89       | 🟡 Medium confidence: Review recommended |
| < 70        | 🔴 Low confidence: Needs manual verification |

---

## Requirements

Install all dependencies:

pip install -r requirements.txt

or manually

python matching.py


This generates:
Matched_Transactions.xlsx
Containing:
- Customer Name
- Customer Contact / Account Reference
- Match Confidence Score
- Match Status

---

## Example Output Summary


Total Transactions Processed: 740
Matches Found: 613
Unmatched / Needs Manual Review: 127

Confidence Score Breakdown:
90–100 : 450 transactions
80–89 : 117 transactions
60–79 : 46 transactions
<60 : 127 transactions


---

## Benefits

| Feature | Benefit |
|--------|---------|
| Automated Matching | Reduces time spent on manual verification |
| Confidence Scoring | Gives transparency on accuracy reliability |
| Adaptable & Reusable | Works for new data without code change |
| Supports Real Bank Narration Noise | Handles spelling variations & short forms |

---

## Author

**Harish**

GitHub Repository:  
https://github.com/HARISHs202003/customer-transaction-matching
