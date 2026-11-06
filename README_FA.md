# کد تشخیص احساس گفتار — نسخهٔ بازبینی‌شده v10

این مخزن شامل کد آموزش، استخراج ویژگی، ارزیابی، پیش‌بینی، آزمون‌ها و نوت‌بوک Colab برای چهار کلاس `neutral`، `happy`، `sad` و `angry` است. راهنمای کامل نصب و اجرا در [README انگلیسی](README.md) آمده است.

دادهٔ خام، وزن‌های آموزش‌دیده، ویژگی‌های آماده، پیش‌بینی‌های ذخیره‌شده و متادیتای اجرای قبلی در این مخزن قرار ندارند. دستورهای راهنما یک آزمایش تازه روی دستگاه شما ایجاد می‌کنند. نتایج مقاله متعلق به اجرای آرشیوشده‌اند و با دانلود کد، اجرای تازه‌ای انجام نمی‌شود.

## شروع کار

از Python 3.12 و یک محیط مجازی مستقل استفاده کنید. مجموعه‌های اصلی را از [RAVDESS](https://zenodo.org/records/1188976) و [TESS](https://doi.org/10.5683/SP2/E8H2MF) دریافت کنید و مجوز آن‌ها را رعایت کنید. داده‌های قبلاً تقویت‌شده را جایگزین دادهٔ اصلی نکنید.

```bash
python -m pip install -r requirements.txt
python ser_data_v10.py --ravdess data/RAVDESS --tess data/TESS --out runs/new_features --workers 2
python ser_models_v10.py train --features runs/new_features/features.npz --manifest runs/new_features/manifest.csv --output runs/new_foldA --models svm cnn r0 r1 r2 r3 --seeds 17 42 73 --epochs 25
```

ابتدا نتایج validation را بررسی و انتخاب مدل را نهایی کنید. سپس انتخاب‌ها را قفل و آزمون را اجرا کنید:

```bash
python ser_models_v10.py freeze --output runs/new_foldA
python ser_models_v10.py evaluate --features runs/new_features/features.npz --manifest runs/new_features/manifest.csv --output runs/new_foldA
```

برای هر آزمایش جدید پوشهٔ خروجی تازه بسازید. تنظیم دوبارهٔ مدل پس از دیدن نتایج آزمون، به ارزیابی مستقل تازه نیاز دارد.

## استفاده از مدل ساخته‌شده

```bash
python ser_infer.py recording.wav --model runs/new_foldA/svm/best.joblib --feature-metadata runs/new_features/feature_metadata.json --output prediction.json
python ser_demo.py --model runs/new_foldA/svm/best.joblib --metadata runs/new_features/feature_metadata.json
```

رابط رومیزی به Tkinter و محیط گرافیکی نیاز دارد و یک فایل صوتی کامل را تحلیل می‌کند. این نسخه ادعای تأخیر اندازه‌گیری‌شده برای میکروفون زنده ندارد. امتیاز خروجی مدل، «دقت» یا اطمینان کالیبره‌شده نیست.

## مدل ازپیش‌آموزش‌دیده و Colab

مسیر emotion2vec به محیط مجازی جداگانه، encoder رسمی و آموزش head نیاز دارد؛ دستورهای دقیق در README انگلیسی هستند. ابزار دریافت، شناسهٔ نسخه و checksum مدل را بررسی می‌کند.

[نوت‌بوک Colab](SER_v10_Colab.ipynb) مسیر مدل‌های کوچک CNN، residual و SVM آکوستیکی را پوشش می‌دهد. آزمایش‌های emotion2vec و مدل regularized، اسکریپت‌های جداگانه دارند.

این پروژه روی گفتار بازی‌شدهٔ انگلیسی و تعداد محدودی گوینده بررسی شده است. تقسیم‌بندی گوینده‌ها مستقل است، ولی کلمات و جمله‌ها می‌توانند بین بخش‌ها تکرار شوند. عملکرد روی گفتار طبیعی یا فارسی در این آزمایش تأیید نشده است.

مجوز مجموعه‌های داده و encoder مستقل از کد پروژه است. برای کد پروژه هنوز مجوز سراسری انتخاب نشده است.
