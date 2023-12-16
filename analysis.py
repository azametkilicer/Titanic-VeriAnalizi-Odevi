import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


#titanic veri setini yükleyelim
df_titanic = sns.load_dataset('titanic')


#veri setinin ilk bir kaç satırını kontrol edelim

print (df_titanic.head())

print(df_titanic.describe())

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_titanic['age'])
plt.title('Age Boxplot')

plt.subplot(1,2,2)
sns.boxplot(y=df_titanic['fare'])
plt.title('Fare Boxplot')

plt.show()

# Cinsiyet Dağılımı
sns.countplot(x='sex', data=df_titanic)
plt.title('Gender Distribution')
plt.show()

# Yolcu Sınıfı Dağılımı
sns.countplot(x='class', data=df_titanic)
plt.title('Passenger Class Distribution')
plt.show()

# Yaş Dağılımı
sns.histplot(df_titanic['age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Hayatta Kalma Durumu
sns.countplot(x='survived', data=df_titanic)
plt.title('Survival Status')
plt.show()



# Korelasyon matrisi
correlation_matrix = df_titanic.corr()

# Isı haritası
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()





# Bağımsız ve bağımlı değişkenleri seçme
X = df_titanic[['pclass', 'age', 'fare']]  # Bağımsız değişkenler
y = df_titanic['survived']                 # Bağımlı değişken

# NaN değerleri temizleme
X = X.fillna(X.mean())

# Modeli kurma ve sonuçları gösterme
model = sm.Logit(y, sm.add_constant(X))
result = model.fit()
print(result.summary())
