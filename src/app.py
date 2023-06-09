import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os

# Useful functions


def load_ml_components(fp):
    "Load the ml components to re-use in app"
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object


def receive_Inputs_Process_And_Predict(
    package_transport_tz,
    main_activity,
    tour_arrangement,
    age_group,
    payment_mode,
    country,
    package_accomodation,
    package_transport_int,
    first_trip_tz,
    package_sightseeing,
    travel_with,
    package_insurance,
    info_source,
    package_guided_tour,
    purpose,
    package_food,
    most_impressing,
    night_zanzibar,
    night_mainland,
    total_female,
    total_male,
):
    """receive inputs, process them and Predict using the ML model"""

    df = pd.DataFrame(
        {
            "night_zanzibar": [night_zanzibar],
            "night_mainland": [night_mainland],
            "total_female": [total_female],
            "total_male": [total_male],
            "package_transport_tz": [package_transport_tz],
            "main_activity": [main_activity],
            "tour_arrangement": [tour_arrangement],
            "age_group": [age_group],
            "payment_mode": [payment_mode],
            "country": [country],
            "package_accomodation": [package_accomodation],
            "package_transport_int": [package_transport_int],
            "first_trip_tz": [first_trip_tz],
            "package_sightseeing": [package_sightseeing],
            "travel_with": [travel_with],
            "package_insurance": [package_insurance],
            "info_source": [info_source],
            "package_guided_tour": [package_guided_tour],
            "purpose": [purpose],
            "package_food": [package_food],
            "most_impressing": [most_impressing],
        }
    )
    df.replace("", np.nan, inplace=True)
    print(f"Inputs as DataFrame : \n{df.to_markdown()}")

    X_for_pred = df

    X_for_pred_ok = pd.concat(
        [
            scaler.transform(num_imputer.transform(X_for_pred[num_cols]))
            if len(num_cols) > 0
            else None,
            encoder.transform(cat_imputer.transform(X_for_pred[cat_cols]))
            if len(cat_cols) > 0
            else None,
        ],
        axis=1,
    )

    y_pred = model.predict(X_for_pred_ok)

    print(f"[Info] Prediction as been made and the output looks like that : {y_pred}")

    return y_pred  # f"My prediction of the expected amount : {y_pred[0]}"


# Setup
# variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")

# Execution
ml_components_dict = load_ml_components(fp=ml_core_fp)
print(f"\n[Info] ML components loaded: {list(ml_components_dict.keys())}")

num_cols = ml_components_dict["num_cols"]
cat_cols = ml_components_dict["cat_cols"]
num_imputer = ml_components_dict["num_imputer"]
cat_imputer = ml_components_dict["cat_imputer"]
scaler = ml_components_dict["scaler"]
encoder = ml_components_dict["encoder"]
model = ml_components_dict["model"]

print(f"\n[Info] Categorical columns : {', '.join(num_cols)}")
print(f"\n[Info] Numeric columns : {', '.join(cat_cols)}\n")

# title and description are optional
title = "Tanzania Tourism Income Prediction"
description = "This model predicts the income for the tourism domain in Tanzania regarding the features of tourist groups. Edit values as you wish to predict the expected income that will be generated by a group."

# Interface
inputs = [
    gr.Dropdown(choices.tolist(), elem_id=i)
    for i, choices in enumerate(encoder.categories_)
] + [gr.Number(elem_id=i) for i in range(4)]
demo = gr.Interface(
    receive_Inputs_Process_And_Predict,
    inputs,
    "number",
    title=title,
    description=description,
    examples=[],
)


if __name__ == "__main__":
    demo.launch(debug=True)
