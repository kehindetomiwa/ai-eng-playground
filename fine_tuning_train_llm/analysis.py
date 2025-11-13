# python
# Install required packages (run once in the notebook)
# Uncomment if packages are not installed:
# !pip install pandas numpy matplotlib seaborn plotly openpyxl kaleido

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

# Jupyter inline plotting
%matplotlib inline
sns.set(style="whitegrid")

# 1) Read Excel
file_path = 'private_schools_datails_ondo_state.xlsx'
try:
    df = pd.read_excel(file_path, engine='openpyxl')
except Exception as e:
    raise RuntimeError(f"Failed to read `{file_path}`: {e}")

# 2) Quick inspection
print("Shape:", df.shape)
print("Columns:", list(df.columns))
display(df.head())

# 3) Basic cleaning: normalize column names (strip, lower, replace spaces)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# Example standardized column suggestions (adjust to actual file)
# Common columns you might have: 'school_name', 'lga', 'school_type', 'enrollment', 'fees', 'latitude', 'longitude'
print("Normalized columns:", list(df.columns))

# 4) Handle missing values and types
# Count missing per column
missing = df.isna().sum().sort_values(ascending=False)
print("Missing values per column:\n", missing)

# Convert numeric-like columns to numeric if present
for col in ['enrollment', 'fees', 'students', 'capacity']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Trim strings
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# 5) Aggregations / summaries
# Schools per LGA
if 'lga' in df.columns:
    lga_counts = df['lga'].value_counts().reset_index()
    lga_counts.columns = ['lga', 'count']
    display(lga_counts.head(10))

# Distribution of school types
if 'school_type' in df.columns:
    type_counts = df['school_type'].value_counts().reset_index()
    type_counts.columns = ['school_type', 'count']
    display(type_counts)

# 6) Static plots (matplotlib / seaborn)
plt.figure(figsize=(10,6))
if 'lga' in df.columns:
    top_n = lga_counts.head(15)
    sns.barplot(data=top_n, x='count', y='lga', palette='viridis')
    plt.title('Top 15 LGAs by number of private schools')
    plt.xlabel('Number of schools')
    plt.ylabel('LGA')
    plt.tight_layout()
    plt.show()

if 'school_type' in df.columns:
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, y='school_type', order=df['school_type'].value_counts().index, palette='pastel')
    plt.title('School type distribution')
    plt.xlabel('Count')
    plt.ylabel('School Type')
    plt.tight_layout()
    plt.show()

# Enrollment distribution (if present)
if 'enrollment' in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df['enrollment'].dropna(), bins=30, kde=True, color='teal')
    plt.title('Enrollment distribution')
    plt.xlabel('Enrollment')
    plt.show()

# 7) Interactive Plotly charts
# Bar: schools per LGA (interactive)
if 'lga' in df.columns:
    fig_lga = px.bar(lga_counts.head(30), x='count', y='lga', orientation='h',
                     title='Top 30 LGAs by number of private schools',
                     labels={'count':'Number of schools','lga':'LGA'})
    fig_lga.update_layout(height=600, width=900, yaxis={'categoryorder':'total ascending'})
    fig_lga.show()
    # Save interactive HTML
    fig_lga.write_html("schools_per_lga_ondo.html")

# Pie or donut for school_type
if 'school_type' in df.columns:
    fig_type = px.pie(type_counts, names='school_type', values='count', title='School Type Distribution')
    fig_type.update_traces(textposition='inside', textinfo='percent+label')
    fig_type.show()
    fig_type.write_html("school_type_distribution_ondo.html")

# Geo scatter if latitude & longitude present
if ('latitude' in df.columns or 'lat' in df.columns) and ('longitude' in df.columns or 'lon' in df.columns):
    lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
    lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
    geo_df = df.dropna(subset=[lat_col, lon_col])
    if not geo_df.empty:
        hover_cols = [c for c in ['school_name','lga','school_type','enrollment'] if c in geo_df.columns]
        fig_map = px.scatter_mapbox(
            geo_df,
            lat=lat_col,
            lon=lon_col,
            hover_name=hover_cols[0] if hover_cols else None,
            hover_data=hover_cols[1:] if len(hover_cols)>1 else None,
            color='school_type' if 'school_type' in geo_df.columns else None,
            zoom=8,
            height=700,
            title='Private Schools in Ondo State'
        )
        fig_map.update_layout(mapbox_style='open-street-map')
        fig_map.show()
        fig_map.write_html("private_schools_map_ondo.html")

# 8) Save a static image of a Plotly figure (requires kaleido)
# Example: save `fig_lga` as PNG if it exists
try:
    if 'fig_lga' in locals():
        fig_lga.write_image("schools_per_lga_ondo.png")
    if 'fig_map' in locals():
        fig_map.write_image("private_schools_map_ondo.png")
except Exception as e:
    print("Saving static images failed (kaleido required):", e)

# 9) Save cleaned CSV for later use
cleaned_path = 'private_schools_ondo_cleaned.csv'
df.to_csv(cleaned_path, index=False)
print("Cleaned data saved to:", cleaned_path)