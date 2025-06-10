from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from fuzzywuzzy import fuzz
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
from convertdate import hebrew
import pandas as pd
import numpy as np
import holidays
import re


def drop(df):
    df.dropna(inplace=True)
    df = df[df["כמות"] > 0]
    return df

def convert_date(df):
    df = df.copy()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.dayofweek       # 0=Monday, 6=Sunday

    weekday_map = {0:2,1:3,2:4,3:5,4:6,5:7,6:1}
    df['Is_Weekend'] = df['Weekday'].isin([4,5]).astype(int)
    df['Day_Name']   = df['Weekday'].map(weekday_map)

    columns_order = [
        'Date','Year','Month','Day','Day_Name','Is_Weekend',
        'תאור פריט','הזמנה','כמות','סכום',
    ]

    return df[columns_order].copy()

def key_words():
    main_dishes_keywords = ['ביאנקה', 'היוונית', 'הצמחונית','טונה דה לקס', 'קריביאן', 'פפרוני ספיישל', 'טוליפ', 'סופר פאפא' , 'המומלצת', 'הבשרית', 'קלאסית', 'מרגריטה', 'האיטלקית', 'ספייסי רול', 'מוצרלה סטיקס'
                           , 'לחמעג\'ון', 'פיצה', 'משפחתיות', 'פיצות', 'משפחתית', 'משפחתי', 'אישית', 'איטסיין', 'פריקסה', 'סופר פאפא','מרגריטות', 'צ\'יזי טונה', 'קלמטה גבינות',
                            'ארוחה', 'ארוחות', 'פאפאדיאס','קלאסיות', 'קלאסי', 'הבלקנית', 'הספרדית', 'הצרפתית', 'סלופי', 'מנה', 'בייטס', 'עקיצת הדבורה','דקה']

    side_dishes_keywords = [
        "צ\'יפס", "אצבעות גבינה", "אצבעות", "בייטס", "נאגטס", "כדורי פירה", 'הקשה של הפיצה', 'פפיוני שום פרמזן' , 'טבעות', 'נגיסי', 'כדורי בשר', 'קראסט', 'הוט-דוג',
        "כדורי גבינה", "מוצרלה סטיקס", 'צ\'יזי רול', "פפיוני שום", 'מנות נלוות',"סלט"]

    desserts_keywords = ['קראנצ', 'פיסטוק', 'בייגלה', 'קרמל', 'שוקולד', 'בלונדי', 'ריקוטה', 'עוגיות', 'גלידה', 'גלידות', 'מגנום', 'קינוח', 'וניל', 'דולצה', 'צ\'אנקי', 'בראוניס', 'טופי', 'אוראו', 'מקופלת',
                         'אגוזי', 'קליק', 'קרם', 'נוגט', 'פירות', 'שמנת', 'קינדר', 'רול']

    toppings_keywords = ['זית', 'זיתים ירוקים','תירס', 'אננס', 'בצל', 'פטריות', 'עגבניות', 'גמבה', 'פלפל', 'חלפיניו',  'גבינות',"6 גבינות", 'תוספת', 'ארטישוק', 'בולגרית', 'טונה', 'אנשובי', 'פפרוני', 'מוצרלה', 'פפרוצ\'יני',
                        'עיזים', 'חצי', 'תוס' ,'תוספות']

    sauces_keywords = ["רוטב", "מארז רטבים", "תבלין", 'אורגנו', "רטבים"]

    drinks_keywords = ["סודה","קולה", "מים", 'פחיות',"ספרייט", "פחית", "זירו", "מונסטר", "פריגת", "פיוז טי", "משקה", "שתייה", 'פאנטה', 'בקבוק','שתיה']

    return {
        "Main Dish": main_dishes_keywords,
        "Dessert": desserts_keywords,
        "Topping": toppings_keywords,
        "Sauce": sauces_keywords,
        "Drink": drinks_keywords,
        "Side Dish": side_dishes_keywords,
    }

def extract_quantity_by_keywords(text):
    if pd.isna(text):
        return {}, {}

    text = text.lower()
    keyword_dict = key_words()
    forbidden_units = {"יח", "יחידות", "ליטר", "ל", "מל"}
    size_tokens = {"8", "14", "16"}
    words = text.split()
    quantity_map = {}
    phrase_map = {}

    for i, word in enumerate(words):
        if word.isdigit():
            num = int(word)
            if num > 150:
                continue
            if str(num) in size_tokens:
                continue
            if (i + 1 < len(words)) and (words[i + 1] in forbidden_units):
                continue

            for offset in [-2, -1, 1, 2]:
                idx = i + offset
                if 0 <= idx < len(words):
                    neighbor = words[idx]
                    for category, keywords in keyword_dict.items():
                        if any(kw in neighbor for kw in keywords):
                            quantity_map[category] = quantity_map.get(category, 0) + num
                            phrase_map[category] = f"{num} {neighbor}"

    return quantity_map, phrase_map

def find_phrase_in_text(text, keywords):
    for kw in keywords:
        if kw in text:
            pattern = rf"(\d+\s*[\w\s\-'״״\"׳]*{re.escape(kw)}[\w\s\-'״״\"׳]*)"
            match = re.search(pattern, text)
            if match:
                phrase = match.group(1).strip()
                phrase = re.sub(r'\b\d+\b', '', phrase)
                return phrase.strip()
    return None


def has_separators(text):
    return any(sep in text for sep in ['+', '/', '&', ' ו ', 'ו', ','])

def split_by_category_sequence(text, keyword_dict):
    words = text.lower().split()
    result = []
    current_group = []
    current_category = None

    def detect_category(word):
        for cat, keywords in keyword_dict.items():
            if any(kw in word for kw in keywords):
                return cat
        return None

    for word in words:
        word_cat = detect_category(word)

        if current_group:
            if word_cat != current_category and word_cat is not None:
                result.append(" ".join(current_group))
                current_group = [word]
                current_category = word_cat
            else:
                current_group.append(word)
        else:
            current_group = [word]
            current_category = word_cat

    if current_group:
        result.append(" ".join(current_group))

    return [re.sub(r'\s{2,}', ' ', g.strip()) for g in result if g.strip()]

def extract_main_dish_prefix(text, size_tokens={"8", "14", "16"}):
    text = text.lower()
    separators = {"+", "/", "&", ",", "ו"}
    tokens = text.split()

    for i, token in enumerate(tokens):
        token_clean = re.sub(r"[^\d]", "", token)
        if token_clean in size_tokens:
            j = i - 1
            while j >= 0:
                if tokens[j] in separators:
                    break
                j -= 1
            main_dish_words = tokens[j+1 : i+1]
            if not main_dish_words:
                return None, text

            prefix = " ".join(main_dish_words).strip()
            remaining_tokens = tokens[: j+1] + tokens[i+1 :]
            remainder = " ".join(remaining_tokens).strip()
            remainder = re.sub(r"\s{2,}", " ", remainder)
            return prefix, remainder

    return None, text

def split_rows_by_category_and_quantity(df):
    df = df.copy()
    rows = []
    keyword_dict = key_words()
    size_tokens = {"8", "14", "16"}

    for _, row in df.iterrows():
        base_data = row.drop(["תאור פריט", "כמות"], errors="ignore").to_dict()

        fallback_qty = row.get("כמות", 1)
        fallback_qty = int(fallback_qty) if pd.notna(fallback_qty) and str(fallback_qty).isdigit() else 1

        original = row.get("תאור פריט", "")
        original = original if pd.notna(original) else ""
        original_text = original.lower()

        main_dish_prefix, remainder_text = extract_main_dish_prefix(original_text, size_tokens)
        if main_dish_prefix:
            cleaned_prefix = re.sub(r"[\[\]\"\'\+\-\(\)\.,]", "", main_dish_prefix)
            cleaned_prefix = re.sub(r"\s{2,}", " ", cleaned_prefix).strip()

            new_row = base_data.copy()
            new_row["Item Description"] = cleaned_prefix
            new_row["Cleaned Description"] = cleaned_prefix
            new_row["Quantity"] = fallback_qty
            new_row["Category"] = "Main Dish"
            rows.append(new_row)

            original_text = remainder_text
            if not original_text.strip():
                continue

        quantity_map, _ = extract_quantity_by_keywords(original_text)

        #Case 1:
        if has_separators(original_text):
            components = re.split(r'\+|/|&|,|\s+ו(?=[\u0590-\u05FF])', original_text)
            for comp in components:
                comp = comp.strip()
                category = None
                quantity = fallback_qty

                for cat, keywords in keyword_dict.items():
                    if any(kw in comp for kw in keywords):
                        category = cat
                        break

                if category and category in quantity_map:
                    for kw in keyword_dict[category]:
                        pattern = rf"\b(\d+)\s*{re.escape(kw)}|\b{re.escape(kw)}\s*(\d+)\b"
                        match = re.search(pattern, comp)
                        if match:
                            quantity = int(match.group(1) or match.group(2))
                            full_match = match.group(0)
                            cleaned_match = re.sub(rf"\b{quantity}\b", "", full_match).strip()
                            comp = comp.replace(full_match, cleaned_match, 1)
                            break

                cleaned = re.sub(r"[\[\]\"\'\+\-\(\)\.,]", "", comp)
                cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

                new_row = base_data.copy()
                new_row["Item Description"] = cleaned
                new_row["Cleaned Description"] = cleaned
                new_row["Quantity"] = quantity
                new_row["Category"] = category
                rows.append(new_row)

        #Case 2:
        elif quantity_map:
            text_cleaned = original_text

            for category, qty in quantity_map.items():
                for kw in keyword_dict.get(category, []):
                    pattern = rf"\b(\d+)\s*{re.escape(kw)}|\b{re.escape(kw)}\s*(\d+)\b"
                    matches = list(re.finditer(pattern, text_cleaned))
                    for match in matches:
                        number = match.group(1) or match.group(2)
                        if number and int(number) == qty:
                            full_match = match.group(0)
                            cleaned_match = re.sub(rf"\b{number}\b", "", full_match).strip()
                            text_cleaned = text_cleaned.replace(full_match, cleaned_match, 1)
                            break

            used_categories = set()

            for category, qty in quantity_map.items():
                if category in used_categories:
                    continue
                used_categories.add(category)

                phrase = find_phrase_in_text(text_cleaned, keyword_dict.get(category, [])) or text_cleaned
                cleaned = re.sub(rf'\b{qty}\b', '', phrase)
                cleaned = re.sub(r"[\[\]\"\'\+\-\(\)\.,]", "", cleaned)
                cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

                new_row = base_data.copy()
                new_row["Item Description"] = cleaned
                new_row["Cleaned Description"] = cleaned
                new_row["Quantity"] = qty
                new_row["Category"] = category
                rows.append(new_row)

        #Case 3:
        else:
            components = split_by_category_sequence(original_text, keyword_dict)
            for comp in components:
                cleaned = re.sub(r"[\[\]\"\'\+\-\(\)\.,]", "", comp)
                cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

                category = None
                for cat, keywords in keyword_dict.items():
                    if any(kw in cleaned for kw in keywords):
                        category = cat
                        break

                new_row = base_data.copy()
                new_row["Item Description"] = cleaned
                new_row["Cleaned Description"] = cleaned
                new_row["Quantity"] = fallback_qty
                new_row["Category"] = category
                rows.append(new_row)

    df_split = pd.DataFrame(rows)
    df_split = df_split[~df_split["Cleaned Description"].str.startswith("בלי")]
    df_split = df_split[~df_split["Cleaned Description"].str.startswith("שובר")]
    df_split = df_split[~df_split["Cleaned Description"].str.startswith("מבצע")]
    return df_split

def group_similar_descriptions_with_size(descs, sizes, threshold=90, top_k=10):
    groups = {}
    seen = set()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', min_df=1)
    vectors = vectorizer.fit_transform(descs)
    sim_matrix = cosine_similarity(vectors)

    for i in range(len(descs)):
        if i in seen:
            continue
        group = [i]
        seen.add(i)
        for j in sim_matrix[i].argsort()[::-1][1:top_k + 1]:
            if j in seen or sizes[i] != sizes[j]:
                continue
            if fuzz.ratio(descs[i], descs[j]) >= threshold:
                group.append(j)
                seen.add(j)

        group_texts = [descs[k] for k in group]
        canonical = Counter(group_texts).most_common(1)[0][0]
        for k in group:
            groups[descs[k]] = canonical

    return groups

def clean_description_for_similarity(text):
    text = str(text).lower()

    marketing_words = {
        "מבצע", "במבצע", "שובר", "חדש", "הטבה", "חינם", "שח", "₪", "ש\"ח", "מחיר",
        "עם", "בלי", "ללא", "בתוספת", "תוספת", "נוסח", "ועוד", "ומעט", "מעט", "rl"
    }
    for word in marketing_words:
        text = text.replace(word, "")

    typo_corrections = {
        "קלסית": "קלאסית",
        "משפחתיות ": "משפחתית",
        "אישיות": "אישית",
        "קינוחים": "קינוח",
        "צמחוניות": "צמחונית",
        "שתיה": "שתייה",
        "ושתייה": "שתייה",
        "פיצות": "פיצה",
        "זיתים ורים": "זיתים ירוקים",
        "משפחתי ": "משפחתית"}

    for typo, corrected in typo_corrections.items():
        text = text.replace(typo, corrected)

    keep_if_near_units = {"יח", "יחידות", "ליטר", "ל", "מל"}
    always_allowed_numbers = {"8", "14", "16", "0.5", "1", "1.5", "15"}

    tokens = text.split()
    cleaned_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ""
        prev_token = tokens[i - 1] if i > 0 else ""

        if re.fullmatch(r"\d+(\.\d+)?", token):
            keep = (
                    token in always_allowed_numbers or
                    next_token in keep_if_near_units or
                    prev_token in keep_if_near_units)
            if keep:
                cleaned_tokens.append(token)
        else:
            cleaned_tokens.append(token)
        i += 1

    cleaned = " ".join(cleaned_tokens)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned

def extract_pizza_size(text):
    match = re.search(r"\b(8|14|16)\b", text)
    return match.group(0) if match else None

def similar_categories_with_size_check(df, default_threshold=85):
    df = df.copy()
    all_rows = []

    threshold_per_category = {
        "Main Dish": 90,
        "Topping": 87,
        "Dessert": 87,
        "Side Dish": 87,
        "Drink": 90,
        "Sauce": 87
    }

    for category in df["Category"].dropna().unique():
        df_cat = df[df["Category"] == category].copy()

        descs = df_cat["Cleaned Description"].dropna().astype(str).unique()
        descs = sorted(descs)

        sizes = [extract_pizza_size(desc) for desc in descs]
        cleaned_descs = [clean_description_for_similarity(d) for d in descs]

        threshold = threshold_per_category.get(category, default_threshold)
        desc_to_canonical = group_similar_descriptions_with_size(
            cleaned_descs, sizes, threshold=threshold, top_k=10)

        mapping = {}
        for desc in descs:
            cleaned = clean_description_for_similarity(desc)
            canonical_cleaned = desc_to_canonical.get(cleaned, cleaned)
            mapping[desc] = canonical_cleaned

        df_cat["Cleaned Description Normalized"] = df_cat["Cleaned Description"].map(mapping).fillna(
            df_cat["Cleaned Description"])
        df_cat["pizza_size"] = df_cat["Cleaned Description"].apply(extract_pizza_size)

        all_rows.append(df_cat)

    df_final = pd.concat(all_rows, ignore_index=True)
    return df_final


def fill_missing_categories_with_model(df, model, index_to_category, feature_col, threshold=0.9):
    df = df.copy()
    mask = df['Category'].isna()

    if mask.any():
        texts = df.loc[mask, feature_col].astype(str)
        probas = model.predict_proba(texts)
        preds = model.predict(texts)
        max_probas = probas.max(axis=1)

        final_cats = [
            index_to_category[pred] if prob >= threshold else None
            for pred, prob in zip(preds, max_probas)
        ]

        df.loc[mask, 'Category'] = final_cats

    return df

def add_holiday_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    us_holidays = holidays.US()
    df["Is_Christian_Holiday"] = df["Date"].dt.date.apply(lambda x: x in us_holidays)
    df["Christian_Holiday_Name"] = df["Date"].dt.date.apply(lambda x: us_holidays.get(x) if x in us_holidays else None)

    jewish_holidays = {
    "Rosh Hashanah": [(1, 1), (2, 1)],
    "Yom Kippur": [(10, 1)],
    "Sukkot": [(15, 1), (16, 1)],
    "Shemini Atzeret": [(22, 1)],
    "Hanukkah": [(25, 9), (26, 9), (27, 9), (28, 9), (29, 9), (30, 9), (1, 10), (2, 10)],
    "Tu BiShvat": [(15, 11)],
    "Purim": [(14, 12)],
    "Passover": [(15, 1), (16, 1), (21, 1), (22, 1)],
    "Independence Day (Israel)": [(5, 2)],
    "Shavuot": [(6, 3)],
}

    def get_jewish_holiday_info(gdate):
        h_year, h_month, h_day = hebrew.from_gregorian(gdate.year, gdate.month, gdate.day)
        for holiday_name, days in jewish_holidays.items():
            if (h_day, h_month) in days:
                return True, holiday_name
        return False, None

    jewish_flags = df["Date"].dt.date.apply(get_jewish_holiday_info)
    df["Is_Jewish_Holiday"], df["Jewish_Holiday_Name"] = zip(*jewish_flags)

    all_holiday_dates = []
    for year in df["Date"].dt.year.unique():
        for holiday_days in jewish_holidays.values():
            for day, month in holiday_days:
                try:
                    g_date = date(*hebrew.to_gregorian(year, month, day))
                    all_holiday_dates.append(g_date)
                except:
                    pass

    near_holiday_dates = set()
    for d in all_holiday_dates:
        for offset in range(-2, 3):
            near_holiday_dates.add(d + timedelta(days=offset))

    df["Is_Near_Jewish_Holiday"] = df["Date"].dt.date.apply(lambda x: x in near_holiday_dates)

    df["Is_Day_Before_New_Year"] = (df["Date"].dt.month == 12) & (df["Date"].dt.day == 31)

    return df

def encode_features(df):
    df = df.copy()

    df["order"] = pd.factorize(df["הזמנה"])[0]
    df.drop(columns=["הזמנה"], inplace=True)

    codes, uniques = pd.factorize(df["Cleaned Description"])
    df["Clean_Desc_Encoded"] = codes

    desc_mapping = {val: i for i, val in enumerate(uniques)}
    pd.Series(desc_mapping).to_csv("Desc_encoding_map.csv", header=["code"], encoding="utf-8-sig")

    le = LabelEncoder()
    df["Category_Encoded"] = le.fit_transform(df["Category"].astype(str))

    category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    mapping_df = pd.DataFrame(list(category_mapping.items()), columns=["Category", "Code"])
    mapping_df.to_csv("Category_mapping.csv", index=False, encoding="utf-8-sig")

    return df

def add_time_features(df):
    df = df.copy()

    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["Season"] = (df["Month"] % 12 // 3 + 1)

    df["Is_Start_of_Month"] = (df["Day"] <= 3).astype(int)
    df["Is_End_of_Month"] = (df["Day"] >= 28).astype(int)

    df["Day_Name_Sin"] = np.sin(2 * np.pi * df["Day_Name"] / 7)
    df["Day_Name_Cos"] = np.cos(2 * np.pi * df["Day_Name"] / 7)

    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    return df

def add_product_features(df):
    df = df.copy()

    def extract_portion_type(desc):
        if isinstance(desc, str):
            desc = desc.lower()
            if any(word in desc for word in ["משפחתית", "משפחתי", "גדול", "גדולה"]):
                return "משפחתית"
            elif any(word in desc for word in ["אישית", "אישי", "קטן", "קטנה"]):
                return "אישית"
        return "מנה"

    df["Portion_Type"] = df["Cleaned Description"].apply(extract_portion_type)

    agg_features = df.groupby("Clean_Desc_Encoded").agg({
        "Quantity": ["mean", "std"],
        "Date": "nunique"
    })
    agg_features.columns = ["Avg_Quantity_All_Time", "Std_Quantity_All_Time", "Num_Days_Sold"]
    agg_features = agg_features.reset_index()

    agg_features["Popularity_Score"] = agg_features["Avg_Quantity_All_Time"]

    portion_type_map = df.groupby("Clean_Desc_Encoded")["Portion_Type"].first().reset_index()

    df = df.merge(agg_features, on="Clean_Desc_Encoded", how="left")
    df = df.merge(portion_type_map, on="Clean_Desc_Encoded", how="left", suffixes=("", "_final"))

    df["Portion_Type"] = df["Portion_Type_Final"]
    df.drop(columns=["Portion_Type_Final"], inplace=True)

    return df

def encode_holiday_and_portion(df):
    df = df.copy()

    # Fill missing values
    df["Jewish_Holiday_Name"] = df["Jewish_Holiday_Name"].fillna("none")
    df["Christian_Holiday_Name"] = df["Christian_Holiday_Name"].fillna("none")
    df["Portion_Type"] = df["Portion_Type"].fillna("none")

    # Create label encoders
    jewish_encoder = LabelEncoder()
    christian_encoder = LabelEncoder()
    portion_encoder = LabelEncoder()

    # Encode columns
    df["Encoded_Jewish_Holiday"] = jewish_encoder.fit_transform(df["Jewish_Holiday_Name"])
    df["Encoded_Christian_Holiday"] = christian_encoder.fit_transform(df["Christian_Holiday_Name"])
    df["Encoded_Portion_Type"] = portion_encoder.fit_transform(df["Portion_Type"])

    # Create and save mapping files
    jewish_map = pd.DataFrame({
        "Jewish_Holiday_Name": jewish_encoder.classes_,
        "Encoded_Value": jewish_encoder.transform(jewish_encoder.classes_)
    })
    jewish_map.to_csv("Jewish_Map_Path.csv", index=False, encoding="utf-8-sig")

    christian_map = pd.DataFrame({
        "Christian_Holiday_Name": christian_encoder.classes_,
        "Encoded_Value": christian_encoder.transform(christian_encoder.classes_)
    })
    christian_map.to_csv("Christian_Map_Path.csv", index=False, encoding="utf-8-sig")

    portion_map = pd.DataFrame({
        "Portion_Type": portion_encoder.classes_,
        "Encoded_Value": portion_encoder.transform(portion_encoder.classes_)
    })
    portion_map.to_csv("Portion_Map_Path.csv", index=False, encoding="utf-8-sig")

    return df

def clean_final_columns(df):
    df = df.copy()

    df = df.drop(columns=["קטגוריות בתיאור", "Description Categories", "Item Description"], errors="ignore")
    df = df.drop(columns=["Jewish_Holiday_Name", "Christian_Holiday_Name", "Portion_Type", "pizza_size"], errors="ignore")
    df = df.drop(columns=["order" ,"סכום"], errors="ignore")

    return df

def sort_by_date(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")

def prepare_data(df, model, index_to_category):
    df = drop(df)
    df = convert_date(df)
    df = split_rows_by_category_and_quantity(df)
    df = similar_categories_with_size_check(df)
    df = fill_missing_categories_with_model(df, model,index_to_category=index_to_category,
                                            feature_col='Cleaned Description Normalized', threshold=0.9)
    df = add_holiday_features(df)
    df = encode_features(df)
    df = add_time_features(df)
    df = add_product_features(df)
    df = encode_holiday_and_portion(df)
    df = clean_final_columns(df)
    df = sort_by_date(df)

    return df





