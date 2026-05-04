# IEEE-CIS Fraud Detection

## Kaggle-ის კონკურსის მოკლე მიმოხილვა

ეს პროექტი ეფუძნება Kaggle-ის კონკურსს (IEEE-CIS Fraud Detection). კონკურსის მიზანია საბანკო ტრანზაქციებში თაღლითობის გამოვლენა. მონაცემები შედგება ორი ფაილისგან: train_transaction.csv და train_identity.csv, რომლებიც გაერთიანებულია TransactionID-ის მიხედვით. სამიზნე ცვლადია isFraud (0 ან 1). შეფასების მეტრიკაა ROC-AUC.

## ჩვენი მიდგომა პრობლემის გადასაჭრელად

პრობლემის გადასაჭრელად გავტესტეთ რამდენიმე სხვადასხვა მოდელი და შევადარეთ მათი შედეგები. თითოეული მოდელისთვის გავლილი იყო ერთი და იგივე ეტაპები: Cleaning, Feature Engineering, Feature Selection და Training. ყველა ექსპერიმენტი დავალოგეთ MLflow-ზე DagsHub-ის საშუალებით.

---

## რეპოზიტორიის სტრუქტურა


IEEE_Fraud_Detection_AM

    notebooks
        model_experiment_LogisticRegression.ipynb
        model_experiment_XGBoost.ipynb
        model_experiment_LightGBM.ipynb
        model_experiment_RandomForest.ipynb
        model_experiment_AdaBoost.ipynb
        model_inference.ipynb

    data/
        train_transaction.csv
        test_transaction.csv
        train_identity.csv
        sample_submission.csv
        test_identity.csv

    README.md

### ფაილების განმარტება

ყველა მოდელს გააჩნია ერთნაირი სტრუქტურა: გვაქვს Cleaning, Feature Engineering, Feature Selection და Training ნაწილები და კიდე სხვა პატარა კოდის ბლოკები, რომლებიც ნაკლებად საინტერესო ნაწილებია კოდის.

model_experiment_LogisticRegression.ipynb — Logistic Regression მოდელის ექსპერიმენტი
model_experiment_XGBoost.ipynb — XGBoost მოდელის ექსპერიმენტი.
model_experiment_LightGBM.ipynb — LightGBM მოდელის ექსპერიმენტი.
model_experiment_RandomForest.ipynb — Random Forest მოდელის ექსპერიმენტი.
model_experiment_AdaBoost.ipynb — AdaBoost მოდელის ექსპერიმენტი.
model_inference.ipynb — საუკეთესო მოდელის (XGBoost) ჩატვირთვა Model Registry-დან და Kaggle-ის submission ფაილის გენერაცია.

გასათვალისწინებელია ისიც, რომ სამწუხაროდ ზომის გამო ვერ მოვახერხე data ფაილების ატვირთვა github-ზე, არვიცი რატომ იყო ეს პრობლემა, მაგრამ მომიწია მარტო notebooks ფაილების ატვირთვა.


## Feature Engineering

### Nan მნიშვნელობების დამუშავება

- რიცხვითი სვეტებისთვის გამოყენებულია SimpleImputer(strategy="median") — მედიანა უფრო გამძლეა outlier-ების მიმართ ვიდრე საშუალო.
- კატეგორიული სვეტებისთვის გამოყენებულია SimpleImputer(strategy="most_frequent") — ყველაზე ხშირი მნიშვნელობით შევსება.

შეიქმნა 3 ახალი feature:

- TransactionAmt_log — ტრანზაქციის თანხის ლოგარითმი (log1p). ეს feature განსაკუთრებით სასარგებლო იყო Logistic Regression-ისთვის, რადგან ამ სვეტში მონაცემები ძალიან გადახრილი (skewed) იყო.
- hour_of_day — ტრანზაქციის საათი დღე-ღამის განმავლობაში, TransactionDT-დან გამოთვლილი. თაღლითობა უფრო ხშირია ღამით.
- null_count — თითოეული მწკრივის ცარიელი მნიშვნელობების რაოდენობა. მაღალი null_count კორელირებს თაღლითობასთან.

### კატეგორიული ცვლადების რიცხვითში გადაყვანა

- Logistic Regression-ისთვის გამოყენებულია OneHotEncoder(handle_unknown="ignore") — კატეგორიული ცვლადები გარდაიქმნა ბინარულ სვეტებად.
- XGBoost, LightGBM, RandomForest და AdaBoost-ისთვის გამოყენებულია OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) — უფრო სწრაფია ხის სტრუქტურაზე დაფუძნებული მოდელებისთვის.


## Cleaning მიდგომები

- წაიშალა სვეტები სადაც 80%-ზე მეტი მნიშვნელობა იყო NaN.
- წაიშალა TransactionID სვეტი, რადგან ის უნიკალური იდენტიფიკატორია და მოდელისთვის სასარგებლო ინფორმაციას არ შეიცავს.


## Feature Selection

### გამოყენებული მიდგომები

1. მაღალი null-ის მქონე სვეტების წაშლა — წაიშალა სვეტები სადაც 80%-ზე მეტი მნიშვნელობა იყო ცარიელი. ეს ამცირებს ხმაურს მოდელში.

2. კორელაციის მიხედვით სვეტების წაშლა — წაიშალა სვეტები რომელთა კორელაცია სხვა სვეტებთან 0.95-ზე მეტი იყო. ზედმეტად კორელირებული სვეტები ერთსა და იმავე ინფორმაციას შეიცავს და მხოლოდ სიჭარბეს ქმნის.

### შეფასება

Feature Selection-მა შეამცირა სვეტების რაოდენობა და გააუმჯობესა მოდელის სიჩქარე. XGBoost და LightGBM-ისთვის Feature Selection ნაკლებად კრიტიკულია, რადგან ეს მოდელები თავად ირჩევენ მნიშვნელოვან feature-ებს. Logistic Regression-ისთვის კი Feature Selection უფრო მნიშვნელოვანია, რადგან ეს მოდელი ვერ ახერხებს feature-ების ავტომატურ შეფასებას.


## Training

### ტესტირებული მოდელები

მოდელი - Sample Size - AUC

- Logistic Regression - 50,000 - 0.848
- Logistic Regression - 150,000 - 0.838
- AdaBoost - 150,000 - 0.865
- Random Forest - 100,000 - 0.883
- LightGBM - 150,000 - 0.916
- XGBoost - 150,000 - 0.917

- საინტერესოა, რომ Logistic Regression-ის შემთხვევაში sample size-ის 50,000-დან 150,000-მდე გაზრდამ გააუარესა შედეგი (0.848 → 0.839). ჩემი აზრით, ეს იმიტომ მოხდა, რომ sample size-ის გაზრდამ საკმაოდ გაზარდა ხმაური
და პირიქით გააუარესა შედეგები ამ მოდელისთვის.

### Hyperparameter ოპტიმიზაციის მიდგომა

- Logistic Regression: გავტესტე სხვადასხვა C მნიშვნელობები (რეგულარიზაციის პარამეტრი). solver="saga" გამოვიყენე დიდი dataset-ისთვის.
- XGBoost და LightGBM: გამოყენებულია n_estimators=300, learning_rate=0.05, max_depth=6. დაბალი learning rate უკეთესია overfitting-ის საწინააღმდეგოდ.
- Random Forest: გამოყენებულია n_estimators=300, max_depth=15.
- AdaBoost: გამოყენებულია n_estimators=300.

### საბოლოო მოდელის შერჩევის დასაბუთება

საბოლოო მოდელად შეირჩა XGBoost (AUC=0.917) შემდეგი მიზეზების გამო:
- ყველაზე მაღალი AUC
- კარგი ბალანსი სიჩქარესა და სიზუსტეს შორის
- კარგად უმკლავდება გამოტოვებულ მნიშვნელობებს და კატეგორიულ ცვლადებს
- Kaggle-ზე public score: 0.909, private score: 0.894 — რაც ადასტურებს რომ მოდელი კარგად გენერალიზდება

---

## MLflow Tracking

### MLflow ექსპერიმენტების ბმული

ყველა ექსპერიმენტი დალოგილია DagsHub-ზე:
https://dagshub.com/AndriaMakharadze/IEEE_Fraud_Detection_AM

### MLflow ექსპერიმენტების სტრუქტურა

თითოეული მოდელისთვის შეიქმნა ცალკე ექსპერიმენტი, შემდეგი run-ებით:

- Cleaning — გასუფთავების ეტაპის პარამეტრები
- Feature_Engineering — შექმნილი feature-ების სია
- Feature_Selection — წაშლილი სვეტების რაოდენობა
- Training — მოდელის hyperparameter-ები და საბოლოო AUC

### ჩაწერილი მეტრიკები და პარამეტრები

- Cleaning: null_threshold, cols_dropped, cols_remaining
- Feature Engineering: new_features, total_cols_after_eng
- Feature Selection: corr_threshold, cols_dropped_corr, cols_remaining
- Training: n_estimators, learning_rate, max_depth, auc

### საუკეთესო მოდელის შედეგები

- მოდელი: XGBoost
- AUC: 0.917
- Kaggle-ის Public Score: 0.909
- Kaggle-ის Private Score: 0.894