import scanpy as sc
import celltypist
from celltypist import models

file = '/Users/rikac/Documents/comp_genomics/project/mouse_male_gonadal.h5ad'
DIR = '/Users/rikac/Documents/comp_genomics/project/'

adata_female = sc.read(file)

# running marker genes for each adata - baseline

# for female gonadal
adata_female = sc.read(file) # 69709 × 25093

# running marker genes for each adata - baseline

# female gonadal
# pre-processing
# mitochondrial genes, "Mt-" for mouse
adata_female.var["mt"] = adata_female.var_names.str.startswith("MT-")
# ribosomal genes
adata_female.var["ribo"] = adata_female.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata_female.var["hb"] = adata_female.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    adata_female, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

sc.pl.violin(
    adata_female,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)
# filter out cells with less than 100 genes and genes that were expressed in less than 3 cells
sc.pp.filter_cells(adata_female, min_genes=100)
sc.pp.filter_genes(adata_female, min_cells=3)

# doublet detection
sc.pp.scrublet(adata_female, batch_key="sample")
len(adata_female[adata_female.obs['is_doublet'] == 1]) # 591
# remove cells with doublets - 69116 × 25093
adata_female_filtered = adata_female[adata_female.obs['is_doublet'] != 1].copy() 

# normalization
# Saving count data
adata_female_filtered.layers["counts"] = adata_female_filtered.X.copy()
# normalizing to median total counts
sc.pp.normalize_total(adata_female_filtered)
# logarithmize the data
sc.pp.log1p(adata_female_filtered)

# get marker genes
cell_counts = adata_female_filtered.obs['cell_type'].value_counts()
min_cells = 10
valid_cell_types = cell_counts[cell_counts >= min_cells].index

adata_female_filtered = adata_female_filtered[adata_female_filtered.obs['cell_type'].isin(valid_cell_types)].copy()
adata_female_filtered.var_names = adata_female_filtered.var['gene_symbols']

sc.tl.rank_genes_groups(
    adata_female_filtered,
    groupby='cell_type',
    method='wilcoxon',  
    use_raw=False,        
    pts=True    # to get % of cells expressing the gene
)

sc.pl.rank_genes_groups(adata_female_filtered, n_genes=10, sharey=False)

# get the top gene expression plot for each cell type on tSNE embedding
sc.tl.tsne(adata_female_filtered, n_pcs=30)
# plotting
adata_female_filtered.var_names = adata_female_filtered.var['feature_name']
sc.pl.tsne(adata_female_filtered, color='Egfl7', use_raw=False) # repeat for all interested gene names

adata_female_filtered.write_h5ad(DIR+'female_gonadal_preprocessed.h5ad')






































































































