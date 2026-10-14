# Dataset and attribution

Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Citation: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

UCI lists this dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Dataset creators: Tiago Almeida and Jos Hidalgo.

`spam.csv` is a conversion of UCI's tab-separated `SMSSpamCollection` into
UTF-8 CSV with `label,message` columns. The message contents and labels are
preserved. This downloaded UCI copy contains **5,574 rows**: 4,827 ham and 747 spam.
Some Kaggle copies report 5,572 rows; do not treat those copies as identical.
The loader prints the actual number of rows and removes repeated messages before
the train/test split. The original UCI readme is preserved as
`UCI_original_readme.txt`.

Alternative: download a Kaggle `spam.csv` with `v1,v2` columns, then run
`python train.py --data path/to/spam.csv`. Extra Kaggle columns are ignored.
