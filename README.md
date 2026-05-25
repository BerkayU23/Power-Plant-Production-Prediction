# Power-Plant-Production-Prediction
Predicting electrical energy output of a Combined Cycle Power Plant using Machine Learning (XGBoost) & Web API

# ⚡ CCPP Energy Forecasting (Kombine Çevrim Elektrik Santrali Üretim Tahmini)

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange?logo=xgboost)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Bu proje, bir Kombine Çevrim Elektrik Santralinin (CCPP) tam yükte çalışırken üreteceği net elektrik enerjisini (MW cinsinden) çevresel sensör verilerini kullanarak tahmin etmek için geliştirilmiş uçtan uca bir veri madenciliği ve makine öğrenmesi çalışmasıdır.

## 🌍 Günlük Hayatta ve Endüstride Ne İşe Yarar?

Elektrik enerjisi, büyük ölçekte depolanması zor ve maliyetli bir kaynaktır. Bu nedenle, üretimin tüketimle anlık olarak eşleşmesi gerekir. Bu projedeki yüksek doğruluklu tahmin modelinin endüstriyel faydaları şunlardır:

* **Şebeke İstikrarı:** Şebeke operatörleri, santralin önümüzdeki saatlerde ne kadar elektrik üreteceğini kesin olarak bilirse, ulusal şebekedeki dalgalanmaları ve kesintileri önleyebilir.
* **Maliyet ve Ticaret Optimizasyonu:** Santral yöneticileri, üretecekleri enerjiyi önceden tahmin ederek enerji borsalarında daha kârlı satış anlaşmaları yapabilir ve üretim eksikliğinden kaynaklı cezalardan kaçınabilirler.
* **Kestirimci Bakım (Predictive Maintenance):** Modelin tahmin ettiği üretim miktarı ile santralin gerçekte ürettiği miktar arasında anormallikler görülmesi, türbinlerde veya sistemde fiziksel bir arızanın başlangıcına işaret eder.

## 🗂️ Proje Dosya Yapısı

* **`CCPP System Forecaasting.py`**: Veri ön işleme (outlier temizliği, standartlaştırma), 5-Fold Cross-Validation ve Hyperparameter Tuning (XGBoost/LightGBM) işlemlerini içeren ana eğitim betiği.
* **`ccppmodel_test.py`**: Eğitilmiş modeli içe aktararak dışarıdan verilen yeni bir veri seti üzerinde canlı tahminleme yapan test betiği.
* **`CCPP_Systems.pkl`**: Eğitilmiş en iyi XGBoost modelini ve veri ölçeklendiriciyi (`StandardScaler`) barındıran serileştirilmiş model dosyası.
* **`ccpptestdata.csv`**: Modelin performansını denemek için ayrılmış, doğru formatta (AT, V, AP, RH) hazırlanmış test veri seti.
* **`ccpp.py`**: Tahmin modelini dış dünyaya açan backend (API) entegrasyon dosyası.
* **`index.html`**: Kullanıcıların çevresel değerleri girerek anlık tahmin sonuçlarını görebileceği web arayüzü.

## ⚙️ Kullanılan Teknolojiler ve Yöntemler

* **Veri İşleme:** Pandas, NumPy
* **Makine Öğrenmesi Yöntemleri:** Multiple Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, LightGBM, XGBoost
* **Model Optimizasyonu:** IQR ile Outlier Tespiti, K-Fold Cross Validation, RandomizedSearchCV
* **Deployment (Dağıtım):** Pickle, HTML, Python Web API

## 🚀 Nasıl Çalıştırılır?

Projeyi kendi bilgisayarınızda test etmek için şu adımları izleyebilirsiniz:

1. Depoyu bilgisayarınıza klonlayın:
2. Gerekli kütüphanelerin yüklü olduğundan emin olun:
      pip install pandas numpy scikit-learn xgboost lightgbm
3.Modeli doğrudan terminal üzerinden test etmek için:
      python "ccppmodel_test.py"
4. Web arayüzünü başlatmak için: ccpp.py dosyasını çalıştırın ve terminalde beliren yerel sunucu adresine (örn: localhost:5000) veya doğrudan index.html dosyasına tarayıcınızdan gidin.
   git clone [https://github.com/KULLANICI_ADIN/CCPP-Energy-Forecasting.git](https://github.com/KULLANICI_ADIN/CCPP-Energy-Forecasting.git)
