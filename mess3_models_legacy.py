#%%

import pandas as pd
metadata_path = "/mnt/extraspace/clin/records/metadata.parquet"
data = pd.read_parquet(metadata_path)

#%%
data