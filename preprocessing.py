from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
from convertdate import hebrew
import pandas as pd
import numpy as np
import holidays
import re

def drop(df):
    
    df.dropna(inplace=True)
    df = df[df["כמות"] >= 0]
    return df
    
def convert_date(df):
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    weekday_map = {
    0: 2,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 1}
    
    df['Year'] = df['Date'].dt.year
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday

    df['Is_Weekend'] = df['Weekday'].isin([4, 5]).astype(int)
    df['Day_Name'] = df['Weekday'].map(weekday_map)
    
    columns_order = [
    'Date',
    'Year',
    'Month',
    'Day',
    'Day_Name',
    'Is_Weekend',
    'תאור פריט',    
    'הזמנה',    
    'כמות', 
    'סכום',]    

    df_all_years = df[columns_order].copy()
    return df_all_years

def key_words():
    main_dishes_keywords = ['ביאנקה', 'טוסקנית', 'היוונית', 'הצמחונית', 'קריביאן', 'פפרוני ספיישל', 'טוליפ', 'סופר פאפא' , 'המומלצת', 'הבשרית', 'קלאסית', 'מרגריטה', 'האיטלקית', 'ספייסי רול', 'מוצרלה סטיקס', 'הקשה של הפיצה',
                            'פפיוני שום פרמזן', 'טבעות', 'לחמעג\'ון', 'צ\'יזי רול', 'נאגטס', 'נגיסי', 'פיצה', 'פיצות', 'משפחתית', 'משפחתי', 'אישית', 'איטסיין', 'פריקסה', 'סופר פאפא', 'קראסט', 'כדורי פירה', 'הוט-דוג', 'כדורי בשר',
                            'ארוחה', 'ארוחות', 'פאפאדיאס', 'קלאסי', 'הבלקנית', 'הספרדית', 'הצרפתית', 'סלופי', 'מנה', 'ציפס', '8', '14', '16', 'בייטס', 'עקיצת הדבורה']

    desserts_keywords = ['קראנצ', 'פיסטוק', 'בייגלה', 'קרמל', 'שוקולד', 'בלונדי', 'ריקוטה', 'עוגיות', 'גלידה', 'גלידות', 'מגנום', 'קינוח', 'וניל', 'דולצה', 'צ\'אנקי', 'בראוניס', 'טופי', 'אוראו', 'מקופלת',
                         'אגוזי', 'קליק', 'קרם', 'נוגט', 'פירות', 'שמנת', 'קינדר', 'רול']

    toppings_keywords = ['זית', 'תירס', 'אננס', 'בצל', 'פטריות', 'עגבניות', 'גמבה', 'פלפל', 'חלפיניו', 'גבינה', 'גבינות', 'תוספת', 'ארטישוק', 'בולגרית', 'טונה', 'אנשובי', 'פפרוני', 'מוצרלה', 'פפרוצ\'יני',
                         'קלמטה', 'עיזים', 'חצי', 'תוספות']

    sauces_keywords = ["רוטב", "מארז רטבים", "תבלין", 'אורגנו']

    drinks_keywords = ["קולה", "מים", "ספרייט", "פחית", "זירו", "מונסטר", "פריגת", "פיוז טי", "משקה", "שתייה", 'פאנטה', 'שתיה']
    
    return {
        "Main Dish": main_dishes_keywords,
        "Dessert": desserts_keywords,
        "Topping": toppings_keywords,
        "Sauce": sauces_keywords,
        "Drink": drinks_keywords
    }
    
def split_category(df):
    
    keyword_dict = key_words()
 
    def categorize_item(description):
        if pd.isna(description):
            return "Other"
        desc = description.lower()
        for category, keywords in keyword_dict.items():
            if any(word in desc for word in keywords):
                return category
        return "Other"

    def split_description(text):
        if pd.isna(text):
            return []
        text = text.lower()
        return re.split(r'\+|/|&| ו ', text)

    def categorize_components(text):
        components = split_description(text)
        categories = set()
        for comp in components:
            cat = categorize_item(comp.strip())
            if cat != "Other":
                categories.add(cat)
        return list(categories) if categories else ["Other"]

    df = df.copy()
    df["Description Categories"] = df["תאור פריט"].apply(categorize_components)
    return df

def extract_quantity_by_keywords(text):
    if pd.isna(text):
        return {}

    text = text.lower()
    result = {}

    category_keywords = key_words()

    for category, keywords in category_keywords.items():
        for kw in keywords:
            pattern = rf"(\d+)\s*{re.escape(kw)}"
            matches = re.findall(pattern, text)
            for m in matches:
                num = int(m)
                result[category] = result.get(category, 0) + num

    return result

def find_phrase_in_text(text, keywords):
    for kw in keywords:
        if kw in text:
            pattern = rf"(\d+\s*[\w\s\-'״״\"׳]*{re.escape(kw)}[\w\s\-'״״\"׳]*)"
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
    return None

def split_rows_by_category_and_quantity(df):
    
    def clean_description_keep_numbers(text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r"[\[\]\"\'\+\-\(\)\.,]", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    rows = []
    keyword_dict = key_words()

    for _, row in df.iterrows():
        base_data = row.drop(["כמות בפועל", "תאור פריט", "כמות"]).to_dict()
        original_text = row["תאור פריט"].lower() if pd.notna(row["תאור פריט"]) else ""
        quantity_dict = row.get("כמות בפועל", {})
        default_qty = row["כמות"]
        categories_in_desc = row.get("קטגוריות בתיאור")

        if isinstance(categories_in_desc, list) and len(categories_in_desc) == 1:
            category = categories_in_desc[0]
            qty = default_qty

            new_row = base_data.copy()
            new_row["Item Description"] = row["תאור פריט"]
            new_row["Cleaned Description"] = clean_description_keep_numbers(row["תאור פריט"])
            new_row["Category"] = category
            new_row["Quantity"] = qty
            rows.append(new_row)

        elif isinstance(quantity_dict, dict) and quantity_dict:
            for category, qty in quantity_dict.items():
                new_row = base_data.copy()

                keywords = keyword_dict.get(category)
                phrase = find_phrase_in_text(original_text, keywords) if keywords else None

                item_text = phrase if phrase else row["תאור פריט"]
                new_row["Item Description"] = item_text
                new_row["Cleaned Description"] = clean_description_keep_numbers(item_text)
                new_row["Category"] = category
                new_row["Quantity"] = qty
                rows.append(new_row)

        else:
            new_row = base_data.copy()
            new_row["Item Description"] = row["תאור פריט"]
            new_row["Cleaned Description"] = clean_description_keep_numbers(row["תאור פריט"])
            new_row["Quantity"] = default_qty
            new_row["Category"] = None
            rows.append(new_row)

    df_split = pd.DataFrame(rows)

    df_split = df_split[~df_split["Cleaned Description"].str.startswith("בלי")]

    return df_split

def add_holiday_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    us_holidays = holidays.US()
    df["is_christian_holiday"] = df["Date"].dt.date.apply(lambda x: x in us_holidays)
    df["christian_holiday_name"] = df["Date"].dt.date.apply(lambda x: us_holidays.get(x) if x in us_holidays else None)
    
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
    df["is_jewish_holiday"], df["jewish_holiday_name"] = zip(*jewish_flags)

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

    df["is_near_jewish_holiday"] = df["Date"].dt.date.apply(lambda x: x in near_holiday_dates)

    df["is_day_before_new_year"] = (df["Date"].dt.month == 12) & (df["Date"].dt.day == 31)

    return df

def encode_features(df):
    df = df.copy()

    df["order"] = pd.factorize(df["הזמנה"])[0]
    df.drop(columns=["הזמנה"], inplace=True)

    codes, uniques = pd.factorize(df["Cleaned Description"])
    df["clean_desc_encoded"] = codes

    desc_mapping = {val: i for i, val in enumerate(uniques)}
    pd.Series(desc_mapping).to_csv("desc_encoding_map.csv", header=["code"], encoding="utf-8-sig")

    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["Category"].astype(str)) 

    category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    mapping_df = pd.DataFrame(list(category_mapping.items()), columns=["Category", "Code"])
    mapping_df.to_csv("category_mapping.csv", index=False, encoding="utf-8-sig")

    return df

def add_time_features(df):
    df = df.copy()

    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["Season"] = (df["Month"] % 12 // 3 + 1)

    df["is_start_of_month"] = (df["Day"] <= 3).astype(int)
    df["is_end_of_month"] = (df["Day"] >= 28).astype(int)

    df["Day_Name_sin"] = np.sin(2 * np.pi * df["Day_Name"] / 7)
    df["Day_Name_cos"] = np.cos(2 * np.pi * df["Day_Name"] / 7)

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

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

    df["portion_type"] = df["Cleaned Description"].apply(extract_portion_type)

    agg_features = df.groupby("clean_desc_encoded").agg({
        "כמות": ["mean", "std"],
        "Date": "nunique"
    })
    agg_features.columns = ["avg_quantity_all_time", "std_quantity_all_time", "num_days_sold"]
    agg_features = agg_features.reset_index()

    agg_features["popularity_score"] = agg_features["avg_quantity_all_time"]

    portion_type_map = df.groupby("clean_desc_encoded")["portion_type"].first().reset_index()

    df = df.merge(agg_features, on="clean_desc_encoded", how="left")
    df = df.merge(portion_type_map, on="clean_desc_encoded", how="left", suffixes=("", "_final"))

    df["portion_type"] = df["portion_type_final"]
    df.drop(columns=["portion_type_final"], inplace=True)

    return df

def encode_holiday_and_portion(df):
    df = df.copy()

    # Fill missing values
    df["jewish_holiday_name"] = df["jewish_holiday_name"].fillna("none")
    df["christian_holiday_name"] = df["christian_holiday_name"].fillna("none")
    df["portion_type"] = df["portion_type"].fillna("none")

    # Create label encoders
    jewish_encoder = LabelEncoder()
    christian_encoder = LabelEncoder()
    portion_encoder = LabelEncoder()

    # Encode columns
    df["encoded_jewish_holiday"] = jewish_encoder.fit_transform(df["jewish_holiday_name"])
    df["encoded_christian_holiday"] = christian_encoder.fit_transform(df["christian_holiday_name"])
    df["encoded_portion_type"] = portion_encoder.fit_transform(df["portion_type"])

    # Create and save mapping files
    jewish_map = pd.DataFrame({
        "jewish_holiday_name": jewish_encoder.classes_,
        "encoded_value": jewish_encoder.transform(jewish_encoder.classes_)
    })
    jewish_map.to_csv("jewish_map_path.csv", index=False, encoding="utf-8-sig")

    christian_map = pd.DataFrame({
        "christian_holiday_name": christian_encoder.classes_,
        "encoded_value": christian_encoder.transform(christian_encoder.classes_)
    })
    christian_map.to_csv("christian_map_path.csv", index=False, encoding="utf-8-sig")

    portion_map = pd.DataFrame({
        "portion_type": portion_encoder.classes_,
        "encoded_value": portion_encoder.transform(portion_encoder.classes_)
    })
    portion_map.to_csv("portion_map_path.csv", index=False, encoding="utf-8-sig")

    return df

def clean_final_columns(df):
    df = df.copy()

    df = df[~df["תאור פריט"].str.contains("שובר", na=False)]

    df = df.drop(columns=["תאור פריט"], errors="ignore")
    df = df.drop(columns=["קטגוריות בתיאור", "Category", "Cleaned Description"], errors="ignore")
    df = df.drop(columns=["jewish_holiday_name", "christian_holiday_name", "portion_type"], errors="ignore")
    df = df.drop(columns=["order", "סכום"], errors="ignore")   

    return df

def sort_by_date(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_sorted = df.sort_values("Date")
    return df_sorted



  

