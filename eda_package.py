import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr, kendalltau, spearmanr, pearsonr
from IPython.display import display
from sklearn.preprocessing import OrdinalEncoder # type: ignore
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report # type: ignore

# 1. Data Exploration
def data_explore(df):
    # Display DataFrame info
    print("=== DataFrame Info ===")
    df.info() 
    print()

    # Count duplicates and total rows
    duplicates = df.duplicated().sum()
    total_rows = len(df)
    percentage_dup = duplicates / len(df) * 100
    percentage_rows = 100

    # DataFrame showing missing values per column
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing Value Count']

    permiss_val = (missing['Missing Value Count'] / total_rows) * 100
    permiss_val = permiss_val.reset_index(drop=True)
    missing['Missing Value Percentage'] = permiss_val
    missing['Missing Value Count'] = missing.apply(
        lambda row: f"{row['Missing Value Count']} ({row['Missing Value Percentage']:.2f}%)", axis=1)
    missing.drop(columns=['Missing Value Percentage'], inplace=True)

    # DataFrame showing unique value counts per column
    unique_counts = df.nunique().reset_index()
    unique_counts.columns = ['Column', 'Unique Value Count']
    
    # DataFrame showing unique items per column
    listItemUnique = []
    for col in df.columns:
        listItemUnique.append([col, df[col].unique().tolist()])
    unique_items = pd.DataFrame(listItemUnique, columns=['Column', 'Unique Items'])

    # DataFrame for duplicate rows and total rows
    dup = pd.DataFrame({
        'Category': ['Duplicate Rows Count', 'Total Rows Count'],
        'Count': [duplicates, total_rows],
        'Percentage': [percentage_dup, percentage_rows]
    })

    # Merge missing and unique_counts based on 'Column'
    summary = pd.merge(missing, unique_counts, on='Column')
    summary = pd.merge(summary, unique_items, on='Column')

    print("\n=== Missing & Unique Values ===")
    display(summary)

    print("\n=== Duplicate Values & Total Rows ===")
    display(dup)

# 2. Descriptive Statistics (Central Tendency)
def descriptive_statistics(df):
    """
    Calculates descriptive statistics such as mean, median, standard deviation, max, min, and quartiles.
    Also includes skewness and kurtosis.
    """
    # Loop through each numeric column in the dataframe
    for col in df.select_dtypes(exclude=object).columns:  # Only for numeric columns
        print(f"\nDescriptive Statistics for column ====> {col}")
        print(f"Mean                         : {df[col].mean():,.2f}")
        print(f"Median                       : {df[col].median():,.2f}")
        print(f"Mode                         : {df[col].mode()[0]:,.2f}")
        print(f"Standard Deviation           : {df[col].std():,.2f}")
        
        # Calculate and display range (max - min)
        range_value = df[col].max() - df[col].min()
        print(f"Range                         : {range_value:,.2f}")
        
        # Skewness and Kurtosis
        print(f"Skewness                      : {df[col].skew():.2f}")
        print(f"Kurtosis                      : {df[col].kurt():.2f}")
        print(f"Minimum Value (Min)          : {df[col].min():.2f}")
        print(f"Quartile 1 Distribution       : {df[col].quantile(0.25):.2f}")
        print(f"Quartile 2 Distribution       : {df[col].quantile(0.50):.2f}")
        print(f"Quartile 3 Distribution       : {df[col].quantile(0.75):.2f}")
        print(f"Maximum Value (Max)          : {df[col].max():.2f}")

# 3. Plot Distribution
def plot_distributions(df, columns, plot_type='categorical', kde=False, n_cols=3):
    """
    Function to plot distributions of categorical and numeric variables.
    
    Parameters:
    df: DataFrame
        DataFrame containing the data.
    columns: list
        List of columns to be analyzed.
    plot_type: str, optional, default='categorical'
        Type of plot to create, can be 'categorical' or 'numeric'.
    kde: bool, optional, default=False
        If True, adds KDE to the numeric plot.
    n_cols: int, optional, default=3
        Number of columns in the plot layout (number of plots per row).
    """
    
    n_vars = len(columns)
    n_rows = (n_vars + n_cols - 1) // n_cols  # Calculate number of rows for plots
    
    fig = plt.figure(figsize=(20, 4*n_rows))
    
    # Loop through each selected column
    for i, var in enumerate(columns, 1):
        plt.subplot(n_rows, n_cols, i)
        
        if plot_type == 'categorical':
            sns.countplot(data=df, x=var)
            plt.title(f'Distribution of {var}')
            plt.xticks(rotation=45)
        
        elif plot_type == 'numeric':
            if kde:
                sns.histplot(data=df, x=var, kde=True)
            else:
                sns.histplot(data=df, x=var)
            plt.title(f'Distribution of {var}')
            plt.xticks(rotation=45)
        
        else:
            raise ValueError("plot_type must be 'categorical' or 'numeric'")
    
    plt.tight_layout()
    plt.show()

# 4. Check Outliers
def check_outlier(X_train_num, plot=True):
    """
    Menghitung batas bawah, batas atas, dan persentase outlier untuk fitur numerik.
    Juga menampilkan plot distribusi tiap fitur dengan batas outlier.

    Parameters:
    - X_train_num: DataFrame berisi fitur numerik dari data training.
    - plot: Boolean, jika True maka akan memunculkan plot distribusi setiap fitur.

    Returns:
    - DataFrame dengan kolom:
        'column', 'Skewness Value', 'Distribusi', 
        'lower_boundary', 'upper_boundary', 'percentage_total_outlier'
    """

    column = []
    skew_vals = []
    distribusi = []
    lower_bound = []
    upper_bound = []
    percent_total_outlier = []

    for i, col in enumerate(X_train_num.columns):
        skew_val = round(X_train_num[col].skew(), 1)
        distrib = 'normal' if -0.5 <= skew_val <= 0.5 else 'skewed'
        
        # Hitung batas bawah & atas berdasarkan distribusi
        if distrib == 'skewed':
            Q1 = X_train_num[col].quantile(0.25)
            Q3 = X_train_num[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
        else:
            mean = X_train_num[col].mean()
            std = X_train_num[col].std()
            lower = mean - 3 * std
            upper = mean + 3 * std

        # Hitung persentase outlier
        n_upper = (X_train_num[col] > upper).sum()
        n_lower = (X_train_num[col] < lower).sum()
        outlier_percent = ((n_upper + n_lower) / len(X_train_num)) * 100

        # Simpan semua nilai
        column.append(col)
        skew_vals.append(skew_val)
        distribusi.append(distrib)
        lower_bound.append(round(lower, 2))
        upper_bound.append(round(upper, 2))
        percent_total_outlier.append(outlier_percent)

        # Plot distribusi dan batas
        if plot:
            plt.figure(figsize=(8, 2))
            sns.boxplot(x=X_train_num[col], color='skyblue')
            plt.axvline(lower, color='green', linestyle='--', label='Lower Bound')
            plt.axvline(upper, color='red', linestyle='--', label='Upper Bound')
            plt.title(f'Boxplot Fitur: {col} (Skewness: {skew_val})')
            plt.xlabel(col)
            plt.legend()
            plt.tight_layout()
            plt.show()


    # Buat DataFrame hasil
    outliers = pd.DataFrame({
        'column': column,
        'Skewness Value': skew_vals,
        'Distribusi': distribusi,
        'lower_boundary': lower_bound,
        'upper_boundary': upper_bound,
        'percentage_total_outlier (%)': percent_total_outlier
    })

    return outliers

# 5. Correlation Analysis
def correlation_analysis(df, nilai_skew=0.5):
    """
    Menghitung dan memvisualisasikan korelasi antar fitur numerik.
    
    - Pearson: jika semua kolom numerik berskew rendah (< nilai_skew)
    - Spearman: jika ada kolom dengan skew tinggi
    - Point-Biserial: jika disediakan target_col biner
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset input
    target_col : str or None
        Nama kolom target biner (default None)
    nilai_skew : float
        Batas ambang skewness (default 0.5)
    alpha : float
        Level signifikansi untuk Point-Biserial correlation (default 0.05)
    """

    # Pilih kolom object
    df_obj = df.select_dtypes(include='object')    
    # Pilih kolom numerik saja
    df_num = df.select_dtypes(include='number')

    # Deteksi kolom normal dan skewed berdasarkan skewness
    normal_cols = []
    skewed_cols = []
    object_cols = []

    for col in df_num.columns:
        skew_val = df[col].skew()
        if abs(skew_val) < nilai_skew:
            normal_cols.append(col)
        else:
            skewed_cols.append(col)
    
    for col in df_obj.columns:
        object_cols.append(col)

    print(f"Normal Distribution Columns   : {normal_cols if normal_cols else '-- Tidak ada kolom normal --'}")
    print(f"Skewed Distribution Columns   : {skewed_cols if skewed_cols else '-- Tidak ada kolom skewed --'}")
    print(f"Object Columns                : {object_cols if object_cols else '-- Tidak ada kolom object --'}")
    print()

    # Tentukan metode korelasi utama
    if len(normal_cols) > 0:
        method = 'pearson'
        print(f"Using correlation method      : {method.upper()} ===> {normal_cols}")
        corr_matrix_pearson = df_num.corr(method=method)
        
        # Visualisasi korelasi antar fitur numerik
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        # plt.xticks(rotation=45)
        plt.title(f"Correlation Matrix ({method.capitalize()})")
        plt.tight_layout()
        plt.show()
        pval_pearson = pd.DataFrame(np.ones_like(corr_matrix_pearson), columns=df_num.columns, index=df_num.columns)

        for i in normal_cols:
            for j in normal_cols:
                if i != j:
                    _, pval = pearsonr(df[i], df[j])
                    pval_pearson.loc[i, j] = pval
                else:
                    pval_pearson.loc[i, j] = 0.0  # diagonal

        # Menyertakan p-value pada matrix signifikansi
        signif_pearson = pval_pearson < 0.05
        print(f"\nSignificance Matrix (p < 0.05) - Pearson:")
        signif_pearson_with_pval = signif_pearson.astype(str) + ' (' + pval_pearson.round(4).astype(str) + ')'
        display(signif_pearson_with_pval)


    if len(skewed_cols) > 0:
        method = 'spearman'
        print(f"Using correlation method      : {method.upper()} ===> {skewed_cols}")
        corr_matrix_spearman = df_num.corr(method=method)

        # Visualisasi korelasi antar fitur numerik
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix_spearman, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        # plt.xticks(rotation=45)
        plt.title(f"Correlation Matrix ({method.capitalize()})")
        plt.tight_layout()
        plt.show()

        pval_spearman = pd.DataFrame(np.ones_like(corr_matrix_spearman), columns=df_num.columns, index=df_num.columns)

        for i in skewed_cols:
            for j in skewed_cols:
                if i != j:
                    _, pval = spearmanr(df[i], df[j])
                    pval_spearman.loc[i, j] = pval
                else:
                    pval_spearman.loc[i, j] = 0.0

        # Menyertakan p-value pada matrix signifikansi
        signif_spearman = pval_spearman < 0.05
        print(f"\nSignificance Matrix (p < 0.05) - Spearman:")
        signif_spearman_with_pval = signif_spearman.astype(str) + ' (' + pval_spearman.round(4).astype(str) + ')'
        display(signif_spearman_with_pval)

    if object_cols:   
        # Encoding
        encoder = OrdinalEncoder()
        df_obj_encoded = df_obj.copy()
        df_obj_encoded[:] = encoder.fit_transform(df_obj)

        method = 'kendall'
        print(f"Using correlation method      : {method.upper()}")
        # Hitung korelasi Kendall's tau manual karena pandas .corr() tidak mendukung kendall untuk DataFrame
        corr_matrix_kendalltau = df_obj_encoded.corr(method=method)
 
        # Visualisasi
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix_kendalltau.astype(float), annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        # plt.xticks(rotation=45)
        plt.title("Correlation Matrix (KENDALL) due to object columns")
        plt.tight_layout()
        plt.show()

        pval_kendall = pd.DataFrame(np.ones_like(corr_matrix_kendalltau), columns=df_obj.columns, index=df_obj.columns)

        for i in object_cols:
            for j in object_cols:
                if i != j:
                    _, pval = kendalltau(df_obj_encoded[i], df_obj_encoded[j])
                    pval_kendall.loc[i, j] = pval
                else:
                    pval_kendall.loc[i, j] = 0.0

        # Menyertakan p-value pada matrix signifikansi
        signif_kendall = pval_kendall < 0.05
        print(f"\nSignificance Matrix (p < 0.05) - Kendall:")
        signif_kendall_with_pval = signif_kendall.astype(str) + ' (' + pval_kendall.round(4).astype(str) + ')'
        display(signif_kendall_with_pval)

# 6. Point-Bisserial Correlation
# Fungsi untuk menghitung Cramer's V
def cramer_v(contingency_table):
    chi2, p_val, dof, ex = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

def correlation_analysis_binary(df, target_col, alpha=0.05, h0=None, h1=None, show=True):
    # Periksa apakah kolom target ada dalam DataFrame
    if target_col not in df.columns:
        print(f"Kolom target '{target_col}' tidak ditemukan.")
        return
    
    target = df[target_col]
    
    # Ubah kolom target menjadi tipe kategori jika belum
    df[target_col] = df[target_col].astype('category')
    
    print(f"\nAnalisis Korelasi terhadap target ===> '{target_col}'")
    
    # === 1. Analisis Point-Biserial ===
    df_num = df.select_dtypes(include='number')  # Memilih kolom numerik
    if df_num.empty:
        print("\nTidak ada kolom numerik untuk analisis Point-Biserial.")
    else:
        pb_results = []
        for col in df_num.columns:
            if col != target_col:
                series = df[col].dropna()  # Menghapus missing values
                aligned_target = target.loc[series.index]
                
                # Pastikan target adalah biner (0 dan 1)
                if aligned_target.nunique() == 2:
                    r_pb, p_val = pointbiserialr(series, aligned_target)
                    signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

                    pb_results.append({
                        "Feature": col,
                        "r_pb": round(r_pb, 3),
                        "p_value": round(p_val, 10),
                        "Significance": signif
                    })
                else:
                    print(f"Kolom {col} tidak memenuhi kriteria target biner untuk Point-Biserial.")

        pb_df = pd.DataFrame(pb_results).sort_values(by='r_pb', ascending=False)
        
        # Tampilkan hasil Point-Biserial
        print("\n=== Hasil Point-Biserial Correlation ===")
        print(pb_df)

        if show:
            if h0 is None or h1 is None:
                for col in df_num.columns:
                    if col != target_col:
                        print(f"\nH0: Tidak ada hubungan antara {target_col} dan {col}.")
                        print(f"H1: Ada hubungan antara {target_col} dan {col}.")
            else:
                print("\n=== Hipotesis yang Diberikan ===")
                print(f"H0: {h0}")
                print(f"H1: {h1}")

            for index, row in pb_df.iterrows():
                if row['p_value'] < alpha:
                    print(f"\nKesimpulan: Ada hubungan antara {target_col} dan {row['Feature']}")
                else:
                    print(f"\nKesimpulan: Tidak ada hubungan antara {target_col} dan {row['Feature']}")

        # Heatmap untuk Point-Biserial Correlation
        heatmap_df = pb_df.set_index('Feature')[['r_pb']]
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_df, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt=".3f", cbar_kws={'label': 'Point-Biserial Correlation (r_pb)'})
        plt.title(f'Point-Biserial Correlation terhadap "{target_col}"', fontsize=12)
        plt.ylabel("Fitur")
        plt.tight_layout()
        plt.show()
    
    # === 2. Analisis Chi-Square ===
    df_cat = df.select_dtypes(include='object')  # Memilih kolom kategorikal
    if df_cat.empty:
        print("\nTidak ada kolom kategorikal untuk analisis Chi-Square.")
    else:
        chi_results = []
        for col in df_cat.columns:
            df[col] = df[col].astype('category')
            
            contingency_table = pd.crosstab(df[col], target)
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                continue  # Skip kolom yang kontingensinya kurang dari 2x2
            chi2, p_val, dof, ex = chi2_contingency(contingency_table)
            signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

            # Menghitung Cramer's V
            cramer_v_value = cramer_v(contingency_table)

            # Menambahkan interpretasi Cramer's V
            if cramer_v_value > 0.25:
                cramer_interpretation = 'Very Strong'
            elif cramer_v_value > 0.15:
                cramer_interpretation = 'Strong'
            elif cramer_v_value > 0.10:
                cramer_interpretation = 'Moderate'
            elif cramer_v_value > 0.05:
                cramer_interpretation = 'Weak'
            else:
                cramer_interpretation = 'No or Very Weak'

            chi_results.append({
                "Feature": col,
                "Chi2": round(chi2, 3),
                "p_value": round(p_val, 10),
                "Significance": signif,
                "Cramer's V": round(cramer_v_value, 3),
                "Interpretation": cramer_interpretation
            })

        chi_df = pd.DataFrame(chi_results).sort_values(by='Chi2', ascending=False)
        
        # Tampilkan hasil Chi-Square
        print("\n=== Hasil Chi-Square Analysis ===")
        print(chi_df)

        if show:
            if h0 is None or h1 is None:
                for col in df_cat.columns:
                    if col != target_col:
                        print(f"\nH0: Tidak ada hubungan antara {target_col} dan {col}.")
                        print(f"H1: Ada hubungan antara {target_col} dan {col}.")
            else:
                print("\n=== Hipotesis yang Diberikan ===")
                print(f"H0: {h0}")
                print(f"H1: {h1}")

            for index, row in chi_df.iterrows():
                if row['p_value'] < alpha:
                    print(f"\nKesimpulan: Ada hubungan antara {target_col} dan {row['Feature']}")
                else:
                    print(f"\nKesimpulan: Tidak ada hubungan antara {target_col} dan {row['Feature']}")

        # Heatmap untuk Chi-Square Analysis
        heatmap_df = chi_df.set_index('Feature')[['Chi2']]
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', fmt=".3f", linewidths=0.5, cbar_kws={'label': 'Chi-Square Statistic'})
        plt.title(f'Chi-Square Analysis terhadap "{target_col}"', fontsize=12)
        plt.ylabel("Fitur")
        plt.tight_layout()
        plt.show()

# 7. Cek persentase missing value pada fitur tertentu
def persentase_missing_value(df_train, df_test, fitur_list):
    # Hitung persentase null di train
    nilainulltrain = (df_train[fitur_list].isnull().sum() / len(df_train)) * 100
    nntr = nilainulltrain.reset_index().rename(columns={'index': 'kolom', 0: 'persentase null train'})
    
    # Hitung persentase null di test
    nilainulltest = (df_test[fitur_list].isnull().sum() / len(df_test)) * 100
    nnts = nilainulltest.reset_index().rename(columns={'index': 'kolom', 0: 'persentase null test'})
    
    # Gabung keduanya
    hasil = pd.merge(nntr, nnts, on='kolom')
    
    return hasil

# 8. Cek persentase dan value tiap kolom
def calculate_value_percentage(df, column, plot=None):
    # Memeriksa apakah kolom ada dalam DataFrame
    if column not in df.columns:
        raise ValueError(f"Kolom '{column}' tidak ditemukan dalam DataFrame.")
    
    # Menghitung jumlah nilai unik untuk kolom yang dipilih
    val_counts = df[column].value_counts()
    
    # Menghitung persentase untuk nilai unik
    per = (val_counts / len(df)) * 100
    
    # Menyimpan hasil perhitungan dalam list
    per_val_col = []
    for value, count in val_counts.items():
        per_val_col.append({
            'Nilai': value,
            'Jumlah': count,
            'Persentase (%)': per[value]
        })
    
    # Membuat DataFrame dari list hasil perhitungan
    pervalcolsum = pd.DataFrame(per_val_col)

    display(pervalcolsum)
    # Visualisasi bar chart jika diinginkan
    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(pervalcolsum['Nilai'].astype(str), pervalcolsum['Jumlah'], color='skyblue')
        plt.title(f'Distribusi Nilai untuk Kolom: {column}')
        plt.xlabel('Nilai')
        plt.ylabel('Jumlah')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    



# 9. Uji Hipotesis t-test (unknown sample)
def t_test_analysis_with_input(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    """
    Fungsi ini melakukan analisis t-test untuk membandingkan rata-rata antara dua kelompok (biner) pada fitur numerik dan target biner,
    dengan inputan manual untuk hipotesis H0 dan H1.
    
    Argumen:
    - df: DataFrame yang berisi data yang ingin dianalisis
    - target_col: Kolom target (harus biner)
    - feature_col: Kolom fitur numerik yang akan diuji
    - alpha: Tingkat signifikansi untuk pengujian hipotesis (default 0.05)
    - h0: Hipotesis Nol (H0), jika tidak diinput, akan menggunakan default
    - h1: Hipotesis Alternatif (H1), jika tidak diinput, akan menggunakan default
    """
    
    # Periksa apakah kolom target dan fitur ada dalam DataFrame
    if target_col not in df.columns or feature_col not in df.columns:
        print(f"Kolom '{target_col}' atau '{feature_col}' tidak ditemukan.")
        return
    
    target = df[target_col]
    feature = df[feature_col]
    
    # Pastikan target adalah biner
    if target.nunique() != 2:
        print(f"Kolom target '{target_col}' bukan biner. Tidak bisa hitung t-test.")
        return

    print(f"\nAnalisis t-test untuk '{feature_col}' terhadap target '{target_col}'")

    # Memisahkan data berdasarkan nilai target (0 atau 1)
    group1 = feature[target == target.unique()[0]]
    group2 = feature[target == target.unique()[1]]

    # Uji t-test antara dua kelompok
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)  # Menggunakan asumsi varian yang tidak sama

    # Menentukan signifikansi
    signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

    # Menampilkan hasil uji t-test
    print("\n=== Hasil t-test ===")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.10f}")
    print(f"Signifikansi: {signif}")

    # Hipotesis: Gunakan input manual jika ada, jika tidak akan menggunakan default
    if h0 is None or h1 is None:
        print("\n=== Hipotesis Default ===")
        print(f"H0: Tidak ada perbedaan rata-rata antara kedua kelompok pada fitur '{feature_col}'")
        print(f"H1: Ada perbedaan rata-rata antara kedua kelompok pada fitur '{feature_col}'")
    else:
        print("\n=== Hipotesis yang Diberikan ===")
        print(f"H0: {h0}")
        print(f"H1: {h1}")

    if p_val < alpha:
        print("\nKesimpulan: Ada hubungan antara", target_col, "dan", feature_col)
    else:
        print("\nKesimpulan: Tidak ada hubungan antara", target_col, "dan", feature_col)
    

# 10. plot_line_relationship
def plot_relationship(dataset, x_col, target_cols, kind='line', figsize=(10, 7), custom_colors=None):
    """
    Fungsi fleksibel untuk memvisualisasi hubungan antara satu kolom X dengan satu atau lebih kolom target Y,
    dalam berbagai jenis plot seaborn: 'line', 'scatter', 'bar', 'hist', 'box', 'violin', 'kde'.
    
    Argumen:
    - dataset: pandas DataFrame
    - x_col: kolom yang akan jadi sumbu X (bisa None untuk plot seperti hist/kde)
    - target_cols: list kolom target Y
    - kind: jenis plot: 'line', 'scatter', 'bar', 'hist', 'box', 'violin', 'kde'
    - figsize: ukuran grafik (default (17, 15))
    - custom_colors: dictionary untuk mengubah warna manual, format {nilai_target: warna}
    """
    # Jika custom_colors diberikan, gunakan warna tersebut, jika tidak, gunakan Set1
    fig, axs = plt.subplots(len(target_cols), 1, figsize=figsize)

    if len(target_cols) == 1:
        axs = [axs]

    for i, target_col in enumerate(target_cols):
        ax = axs[i]
        
        # Jika custom_colors diberikan, gunakan itu, jika tidak, tentukan warna berdasarkan nilai unik pada kolom target
        if custom_colors:
            value_to_color = custom_colors
        else:
            unique_values = dataset[target_col].unique()
            colors = sns.color_palette("Set1", len(unique_values))  # Warna untuk setiap kategori unik
            value_to_color = {value: colors[j] for j, value in enumerate(unique_values)}
        
        if kind == 'line':
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.lineplot(x=subset[x_col], y=subset[target_col], ax=ax, color=value_to_color.get(value, 'gray'), label=value)
        elif kind == 'scatter':
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.scatterplot(x=subset[x_col], y=subset[target_col], ax=ax, color=value_to_color.get(value, 'gray'), label=value)
        elif kind == 'bar':
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.barplot(x=subset[x_col], y=subset[target_col], ax=ax, color=value_to_color.get(value, 'gray'), label=value)
        elif kind == 'hist':
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.histplot(subset[target_col], ax=ax, color=value_to_color.get(value, 'gray'), kde=False, label=value)
        elif kind == 'box':
            sns.boxplot(x=dataset[x_col] if x_col else None, y=dataset[target_col], ax=ax, palette=value_to_color)
        elif kind == 'violin':
            sns.violinplot(x=dataset[x_col] if x_col else None, y=dataset[target_col], ax=ax, palette=value_to_color)
        elif kind == 'kde': 
            for value in dataset[target_col].unique():
                subset = dataset[dataset[target_col] == value]
                sns.kdeplot(data=subset, x=target_col, ax=ax, fill=True, color=value_to_color.get(value, 'gray'), label=value)
        elif kind == 'count':
            sns.countplot(data=dataset, x=target_col, ax=ax, palette=value_to_color)
        else:
            raise ValueError("Jenis plot tidak dikenali. Gunakan 'line', 'scatter', 'bar', 'hist', 'box', 'violin', atau 'kde'.")

        title = f'{target_col}' if kind in ['hist', 'kde'] else f'{x_col} vs {target_col}'
        ax.set_title(f'{kind.title()} Plot: {title}')
        ax.set_xlabel(x_col if x_col else '')
        ax.set_ylabel(target_col)

        # Menambahkan legend
        ax.legend()

    plt.tight_layout()
    plt.show()


# 11. Annova
def anova_analysis_with_input(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    """
    Fungsi ini melakukan analisis ANOVA untuk membandingkan rata-rata antara lebih dari dua kelompok pada fitur numerik dan target kategorikal,
    dengan inputan manual untuk hipotesis H0 dan H1.
    
    Argumen:
    - df: DataFrame yang berisi data yang ingin dianalisis
    - target_col: Kolom target (harus kategorikal dengan lebih dari dua kategori)
    - feature_col: Kolom fitur numerik yang akan diuji
    - alpha: Tingkat signifikansi untuk pengujian hipotesis (default 0.05)
    - h0: Hipotesis Nol (H0), jika tidak diinput, akan menggunakan default
    - h1: Hipotesis Alternatif (H1), jika tidak diinput, akan menggunakan default
    """
    
    # Periksa apakah kolom target dan fitur ada dalam DataFrame
    if target_col not in df.columns or feature_col not in df.columns:
        print(f"Kolom '{target_col}' atau '{feature_col}' tidak ditemukan.")
        return
    
    target = df[target_col]
    feature = df[feature_col]
    
    # Pastikan target memiliki lebih dari dua kategori
    if target.nunique() <= 2:
        print(f"Kolom target '{target_col}' harus memiliki lebih dari dua kategori. Tidak bisa hitung ANOVA.")
        return

    print(f"\nAnalisis ANOVA untuk '{feature_col}' terhadap target '{target_col}'")

    # Memisahkan data berdasarkan kategori pada kolom target
    groups = [feature[target == category] for category in target.unique()]

    # Uji ANOVA antara kelompok-kelompok berdasarkan target
    f_stat, p_val = stats.f_oneway(*groups)

    # Menentukan signifikansi
    signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

    # Menampilkan hasil uji ANOVA
    print("\n=== Hasil ANOVA ===")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"p-value: {p_val:.10f}")
    print(f"Signifikansi: {signif}")

    # Hipotesis: Gunakan input manual jika ada, jika tidak akan menggunakan default
    if h0 is None or h1 is None:
        print("\n=== Hipotesis Default ===")
        print(f"H0: Tidak ada perbedaan rata-rata antara kelompok-kelompok pada fitur '{feature_col}'")
        print(f"H1: Ada perbedaan rata-rata antara kelompok-kelompok pada fitur '{feature_col}'")
    else:
        print("\n=== Hipotesis yang Diberikan ===")
        print(f"H0: {h0}")
        print(f"H1: {h1}")

    if p_val < alpha:
        print("\nKesimpulan: Ada hubungan antara", target_col, "dan", feature_col)
    else:
        print("\nKesimpulan: Tidak ada hubungan antara", target_col, "dan", feature_col)

# 12. Chi-Square Test
def chi_square_analysis(df, target_col, feature_col, alpha=0.05, h0=None, h1=None):
    """
    Fungsi ini melakukan uji Chi-Square untuk menguji apakah ada hubungan antara dua variabel kategorikal
    (misalnya, 'Attrition' dan 'Job Satisfaction').
    
    Argumen:
    - df: DataFrame yang berisi data yang ingin dianalisis
    - target_col: Kolom target (harus kategorikal)
    - feature_col: Kolom fitur kategorikal
    - alpha: Tingkat signifikansi untuk pengujian hipotesis (default 0.05)
    - h0: Hipotesis Nol (H0), jika tidak diinput, akan menggunakan default
    - h1: Hipotesis Alternatif (H1), jika tidak diinput, akan menggunakan default
    """
    
    # Periksa apakah kolom target dan fitur ada dalam DataFrame
    if target_col not in df.columns or feature_col not in df.columns:
        print(f"Kolom '{target_col}' atau '{feature_col}' tidak ditemukan.")
        return
    
    target = df[target_col]
    feature = df[feature_col]
    
    # Membuat tabel kontingensi
    contingency_table = pd.crosstab(target, feature)

    # Uji Chi-Square
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    # Menentukan signifikansi
    signif = "Signifikan" if p_val < alpha else "Tidak signifikan"

    # Menampilkan hasil uji Chi-Square
    print("\n=== Hasil Uji Chi-Square ===")
    print(f"Chi-Square Test Statistic: {chi2_stat:.3f}")
    print(f"p-value: {p_val:.10f}")
    print(f"Signifikansi: {signif}")

    # Menampilkan hipotesis
    if h0 is None or h1 is None:
        # Jika hipotesis tidak diberikan, menggunakan default
        print("\n=== Hipotesis ===")
        print(f"H0: Tidak ada hubungan antara {target_col} dan {feature_col}.")
        print(f"H1: Ada hubungan antara {target_col} dan {feature_col}.")
    else:
        print("\n=== Hipotesis yang Diberikan ===")
        print(f"H0: {h0}")
        print(f"H1: {h1}")

    # Menampilkan kesimpulan
    if p_val < alpha:
        print("\nKesimpulan: Ada hubungan antara", target_col, "dan", feature_col)
    else:
        print("\nKesimpulan: Tidak ada hubungan antara", target_col, "dan", feature_col)
    
    # Visualisasi heatmap Chi-Square (perbandingan observed dan expected)
    plt.figure(figsize=(8, 10))

    # Heatmap Observed Frequencies (Atas)
    plt.subplot(2, 1, 1)
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor='black')
    plt.title("Observed Frequencies (Tabel Kontingensi)", fontsize=14)
    plt.xlabel(feature_col, fontsize=12)
    plt.ylabel(target_col, fontsize=12)

    # Heatmap Expected Frequencies (Bawah)
    plt.subplot(2, 1, 2)
    sns.heatmap(expected, annot=True, fmt=".2f", cmap="Oranges", cbar=False, linewidths=1, linecolor='black')
    plt.title("Expected Frequencies (Dihitung dari Chi-Square)", fontsize=14)
    plt.xlabel(feature_col, fontsize=12)
    plt.ylabel(target_col, fontsize=12)

    plt.tight_layout()
    plt.show()

def evaluate_model_class_report(model, X_train, y_train, X_test, y_test):
    """
    Parameters:
    - model: model yang sudah dilatih (KNN, SVC, Decision Tree, Random Forest, Gradient Boost)
    - X_train: Data fitur untuk training
    - y_train: Data label untuk training
    - X_test: Data fitur untuk testing
    - y_test: Data label untuk testing
    
    Returns:
    - None: Mencetak hasil evaluasi
    """
    # Prediksi hasil model pada data uji dan data latih
    y_pred_tuning_train = model.predict(X_train)
    y_pred_tuning_test = model.predict(X_test)

    # 3. Classification report
    print("=============== Classification Report ===============\n")
    print("Train Data:")
    print(classification_report(y_train, y_pred_tuning_train))
    print('------------------------------------------------------')
    print("Test Data:")
    print(classification_report(y_test, y_pred_tuning_test))

     # 4. Confusion Matrix
    cm_train = confusion_matrix(y_train, y_pred_tuning_train)
    cm_test = confusion_matrix(y_test, y_pred_tuning_test)

    # Plot Confusion Matrix untuk Train Data
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix - Train Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot Confusion Matrix untuk Test Data
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix - Test Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



def analyze_feature_correlations(df, target='Status', alpha=0.05,
                                 numeric_features=[], ordinal_features=[], 
                                 nominal_features=[], binary_features=[]):
    """
    Fungsi untuk menganalisis korelasi antar fitur dengan target.
    
    Parameters:
    df : DataFrame
        DataFrame yang berisi data.
    target : str, optional, default='Status'
        Nama kolom target untuk analisis korelasi.
    alpha : float, optional, default=0.05
        Tingkat signifikansi untuk uji hipotesis.
    numeric_features : list, optional
        List kolom-kolom numerik yang akan dianalisis.
    ordinal_features : list, optional
        List kolom-kolom ordinal yang akan dianalisis.
    nominal_features : list, optional
        List kolom-kolom nominal yang akan dianalisis.
    binary_features : list, optional
        List kolom-kolom binary yang akan dianalisis.
    
    Returns:
    results_df : DataFrame
        DataFrame yang berisi hasil analisis korelasi.
    """
    
    # Inisialisasi list untuk hasil
    results = []
    
    # Fungsi untuk menghitung eta squared (untuk nominal dan binary)
    def calculate_eta_squared(feature, target_series):
        contingency_table = pd.crosstab(feature, target_series)
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        n = len(feature)
        eta_sq = chi2 / (n + chi2)
        return eta_sq, p_value
    
    # Analisis fitur numerik menggunakan korelasi Spearman
    for feature in numeric_features:
        if feature in df.columns:
            correlation, p_value = stats.spearmanr(df[feature], pd.Categorical(df[target]).codes)
            results.append({
                'name_feature': feature,
                'method_corr': 'Spearman',
                'p_value': p_value,
                'significant_or_not': 'Significant' if p_value < alpha else 'Not Significant',
                'corr_value': abs(correlation)
            })
    
    # Analisis fitur ordinal menggunakan korelasi Kendall Tau
    for feature in ordinal_features:
        if feature in df.columns:
            correlation, p_value = stats.kendalltau(pd.Categorical(df[feature]).codes, 
                                                   pd.Categorical(df[target]).codes)
            results.append({
                'name_feature': feature,
                'method_corr': 'Kendall',
                'p_value': p_value,
                'significant_or_not': 'Significant' if p_value < alpha else 'Not Significant',
                'corr_value': abs(correlation)
            })
    
    # Analisis fitur nominal dan binary menggunakan Chi-square dan Eta squared
    for feature in nominal_features + binary_features:
        if feature in df.columns:
            eta_sq, p_value = calculate_eta_squared(df[feature], df[target])
            results.append({
                'name_feature': feature,
                'method_corr': 'Chi-square (Eta-squared)',
                'p_value': p_value,
                'significant_or_not': 'Significant' if p_value < alpha else 'Not Significant',
                'corr_value': eta_sq
            })
    
    # Membuat DataFrame dan mengurutkan berdasarkan nilai korelasi
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('corr_value', ascending=False)
    
    # Membulatkan kolom numerik
    results_df['p_value'] = results_df['p_value'].round(4)
    results_df['corr_value'] = results_df['corr_value'].round(4)
    
    return results_df
