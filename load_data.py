from typing import Union, Literal
from itertools import islice
from datasets import load_dataset
import pandas as pd

def save_dataset(
        train_samples=10_000,
        val_percentage=0.1,
        test_percentage=0.2
    ):

    # Huggingface account & Cli Access is required
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        streaming=True,
        split="train"  # dataset does not have test; we will skip over x samples for the test dataset
    )

    train_data = list(islice(ds, train_samples))
    df_train = pd.DataFrame(train_data)
    df_train.to_parquet("./data/train.parquet")

    val_data = list(islice(ds,
                           train_samples,
                           int(train_samples+(train_samples*val_percentage)))
                   )
    df_val = pd.DataFrame(val_data)
    df_val.to_parquet("./data/val.parquet")

    test_data = list(islice(ds,
                            int(train_samples+(train_samples*val_percentage)),
                            int(train_samples+(train_samples*val_percentage)+(train_samples*test_percentage)))
                    )
    df_test = pd.DataFrame(test_data)
    df_test.to_parquet("./data/test.parquet")


if __name__ == "__main__":
    save_dataset()