# Uyku Verimliliği Tahmini: Regresyon Modelleri Karşılaştırması

Bu proje, bireylerin yaşam tarzı ve uyku alışkanlıklarına ait verileri kullanarak Uyku Verimliliğini (Sleep Efficiency) tahmin etmeyi amaçlayan bir makine öğrenmesi çalışmasıdır. 

Çalışma kapsamında veri ön işleme, aykırı değer analizi gerçekleştirilmiş ve farklı regresyon algoritmaları (OLS, Lasso, RANSAC) kullanılarak aykırı değerlerin (outliers) modeller üzerindeki etkileri incelenmiştir.

## Proje Adımları ve Kullanılan Yöntemler

* Veri Ön İşleme: Zaman değişkenlerinin saat formatına dönüştürülmesi ve kategorik değişkenlerin `LabelEncoder` ile sayısal formata çevrilmesi.
* Eksik Veri Yönetimi: `KNNImputer` ve `RobustScaler` kullanılarak eksik verilerin doldurulması.
* Aykırı Değer Analizi: IQR (%10 - %90) yöntemi ile aykırı değerlerin baskılanması (Capping).
* Modelleme:
  * Standart Doğrusal Regresyon (OLS)
  * Lasso Regresyon (LassoCV ile çapraz doğrulama)
  * RANSAC Regresyon (Aykırı değerlere dirençli modelleme)
* Performans Değerlendirmesi: Modellerin hata kareler ortalaması (MSE), mutlak hata ortalaması (MAE) ve R² skorları üzerinden karşılaştırılması.
* Veri Görselleştirme: Matplotlib ve Seaborn kütüphaneleriyle regresyon doğruları üzerindeki kaldıraç etkisinin (leverage effect), RANSAC inlier/outlier ayrımının ve Lasso hata dağılımının (residual) görselleştirilmesi.

##  Kullanılan Teknolojiler

* Dil: Python 3.x
* Veri İşleme: Pandas, NumPy
* Makine Öğrenmesi: Scikit-Learn
* Görselleştirme: Matplotlib, Seaborn, Missingno

## Kurulum ve Kullanım

Kodu kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Bu depoyu yerel bilgisayarınıza klonlayın:
   ```bash
git clone https://github.com/ebrarcuhadar/regresyon-modelleri-calismalari.git
