## Purpose  
Running the script will train on `train.txt`, validate on `val.txt`, and generate **`predictions.txt`** for the hidden test set.

---

## Quick‑Start (exact grader command)

```bash
# 2 · (optional) create an isolated env
python3 -m venv venv
source venv/bin/activate

# 3 · install dependencies
pip install -r requirements.txt

# 4 · train + predict
python3 classifier.py -d train.txt -v val.txt -t test.txt

## Predictions.txt
- Output follows the below format

``` bash 

<testTweetID> <TAB> <predictedHashtagID>

