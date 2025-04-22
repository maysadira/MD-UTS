

###**UTS MODEL DEPLOYMENT**
#####Putri Maysa Adira
#####2702372826
#####LB09
#####No 3 (Dataset B)


import pandas as pd
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

model_filename = '/content/best_model.pkl'
with open(model_filename, 'rb') as f:
    best_model = pickle.load(f)

label_encoder_filename = '/content/label_encoder.pkl'
with open(label_encoder_filename, 'rb') as f:
    label_encoder = pickle.load(f)

print("Model dan Label Encoder berhasil dimuat!")

user_input = {
    'no_of_adults': [2],
    'no_of_children': [1],
    'no_of_weekend_nights': [2],
    'no_of_week_nights': [3],
    'type_of_meal_plan_Meal Plan 2': 1,
    'type_of_meal_plan_Meal Plan 3': 0,
    'type_of_meal_plan_Not Selected': 0,
    'room_type_reserved_Room_Type 2': 0,
    'room_type_reserved_Room_Type 3': 0,
    'room_type_reserved_Room_Type 4': 1,
    'room_type_reserved_Room_Type 5': 0,
    'room_type_reserved_Room_Type 6': 0,
    'room_type_reserved_Room_Type 7': 0,
    'lead_time': 35,
    'arrival_year': 2018,
    'arrival_month': 5,
    'arrival_date': 12,
    'market_segment_type_Corporate': 0,
    'market_segment_type_Offline': 0,
    'market_segment_type_Online': 1,
    'market_segment_type_Complementary': 0,
    'repeated_guest': 0,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 0,
    'avg_price_per_room': 105.5,
    'no_of_special_requests': 1,

}

user_input['room_type_reserved_Room_Type 1'] = 0
user_input['type_of_meal_plan_Meal Plan 1'] = 0
user_input['market_segment_type_Aviation'] = 0


df_input = pd.DataFrame(user_input)

feature_order = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'lead_time', 'arrival_year', 'arrival_month', 'arrival_date', 'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests', 'type_of_meal_plan_Meal Plan 1', 'type_of_meal_plan_Meal Plan 2', 'type_of_meal_plan_Meal Plan 3', 'type_of_meal_plan_Not Selected', 'room_type_reserved_Room_Type 1', 'room_type_reserved_Room_Type 2', 'room_type_reserved_Room_Type 3', 'room_type_reserved_Room_Type 4', 'room_type_reserved_Room_Type 5', 'room_type_reserved_Room_Type 6', 'room_type_reserved_Room_Type 7', 'market_segment_type_Aviation', 'market_segment_type_Complementary', 'market_segment_type_Corporate', 'market_segment_type_Offline', 'market_segment_type_Online']
df_input = df_input[feature_order]


print("Data yang telah diproses untuk prediksi:")
print(df_input)


prediction = best_model.predict(df_input)[0]


probability = best_model.predict_proba(df_input)[:, 1]

status = "Not Cancelled" if prediction == 0 else "Cancelled"
print(f"Prediction Result: {status}")
print(f"Cancellation Probability: {probability[0]:.2%}")
