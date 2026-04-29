# Rasch Trial Item Analyzer

A beginner-friendly Streamlit web app for analyzing trial exam items using a Rasch-style 1PL model.

Users upload a CSV file, and the app produces:

- Item difficulty estimates
- Infit and Outfit statistics
- Point-measure correlations
- KEEP / REVISE / REMOVE decisions
- Person-item map
- Ability and item difficulty distribution plots
- Downloadable CSV and PDF reports

## CSV Format

Your CSV should look like this:

```csv
person,item_1,item_2,item_3,item_4
P1,1,1,0,0
P2,1,0,0,0
P3,1,1,1,0
```

Rules:

- First column may be a student/person ID.
- Item columns must contain `0` and `1`.
- `1` = correct / success
- `0` = incorrect / failure

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these files:
   - `streamlit_app.py`
   - `requirements.txt`
   - `README.md`
3. Go to Streamlit Community Cloud.
4. Connect your GitHub repository.
5. Choose `streamlit_app.py` as the main file.
6. Deploy.

## Decision Rules

The app uses common beginner screening rules:

- `0.05 < p_correct < 0.95`
- `0.5 <= infit <= 1.5`
- `0.5 <= outfit <= 1.5`
- `point_measure_corr >= 0.20`
- item difficulty roughly between `-3` and `+3` logits

Important: Rasch statistics should support, not replace, expert item review.
