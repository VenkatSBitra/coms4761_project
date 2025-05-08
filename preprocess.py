# %%
import scanpy as sc
import pandas as pd
import numpy as np

# %%
male = sc.read_h5ad("data/mouse_male_gonadal.h5ad")
female = sc.read_h5ad("data/mouse_female_gonadal.h5ad")

# %%
np.unique(male.obs['development_stage'])

# %%
# male.X = male.raw.X
# male.var_names = male.raw.var_names

# female.X = female.raw.X
# female.var_names = female.raw.var_names

# %%
print("Male dataset:")
print("Shape:", male.shape)
print("Data type:", male.X.dtype)
print("Min/Max values:", np.min(male.X), np.max(male.X))
print("")

print("Female dataset:")
print("Shape:", female.shape)
print("Data type:", female.X.dtype)
print("Min/Max values:", np.min(female.X), np.max(female.X))

# %%
sc.pp.filter_cells(male, min_genes=200)
sc.pp.filter_genes(male, min_cells=3)

sc.pp.filter_cells(female, min_genes=200)
sc.pp.filter_genes(female, min_cells=3)

# %%
sc.pp.normalize_total(male, target_sum=1e4)
sc.pp.log1p(male)

sc.pp.normalize_total(female, target_sum=1e4)
sc.pp.log1p(female)

# %%
male.var_names

# %%
female.var_names

# %%
common_genes = male.var_names.intersection(female.var_names)
male_data = male[:, common_genes].copy()
female_data = female[:, common_genes].copy()

# %%
# Filter for male data
cell_counts_male = male_data.obs['cell_type'].value_counts()
valid_cell_types_male = cell_counts_male[cell_counts_male >= 100].index
male_data = male_data[male_data.obs['cell_type'].isin(valid_cell_types_male), :]

# Filter for female data
cell_counts_female = female_data.obs['cell_type'].value_counts()
valid_cell_types_female = cell_counts_female[cell_counts_female >= 100].index
female_data = female_data[female_data.obs['cell_type'].isin(valid_cell_types_female), :]


# %%
common_cell_types = set(male_data.obs['cell_type']).intersection(female_data.obs['cell_type'])
male_data = male_data[male_data.obs['cell_type'].isin(common_cell_types), :]
female_data = female_data[female_data.obs['cell_type'].isin(common_cell_types), :]

# %%
# sc.pp.normalize_total(male_data, target_sum=1e4)
# sc.pp.log1p(male_data)

# sc.pp.normalize_total(female_data, target_sum=1e4)
# sc.pp.log1p(female_data)

# %%
sorted_genes = sorted(male_data.var_names)
male_data = male_data[:, sorted_genes]
female_data = female_data[:, sorted_genes]


# %%
male_df = pd.DataFrame(male_data.X.toarray(), index=male_data.obs_names, columns=sorted_genes)
female_df = pd.DataFrame(female_data.X.toarray(), index=female_data.obs_names, columns=sorted_genes)

# %%
print(male_df.shape)
print(female_df.shape)

# %%
male_data.obs['cell_type']

# %%
female_data.obs['cell_type']

# %%
male_data.write_h5ad('cleaned_data/male_gonadal.h5ad')
female_data.write_h5ad('cleaned_data/female_gonadal.h5ad')

# %%



