import streamlit as st
import pandas as pd
import pickle, os

# Useful functions
def load_ml_components(fp):
    "Load the ml components to re-use in app"
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object

# Setup
## variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")

## Execution
ml_components_dict = load_ml_components(fp=ml_core_fp)

labels = ml_components_dict['labels']
idx_to_labels = {i: l for (i, l) in enumerate(labels)}

end2end_pipeline = ml_components_dict['pipeline']



print(f"\n[Info] Predictable labels: {labels}")
print(f"\n[Info] Indexes to labels: {idx_to_labels}")

print(f"\n[Info] ML components loaded: {list(ml_components_dict.keys())}")

# Image
st.image("https://friendsofthefarm.ca/wp-content/uploads/2022/04/irisgarden.jpg")
# Interface 
st.write(
    """# Iris classification app
This is my interface using Streamlit to classify iris regarding some features.
"""
)


with st.form(key="information", clear_on_submit=True):
    ## Inputs
    sepal_height = st.number_input("Enter sepal height")
    sepal_width = st.number_input("Enter sepal width")

    petal_height = st.number_input("Enter petal height")
    petal_width = st.number_input("Enter petal width")

    ## prediction executed
    if st.form_submit_button("Predict"):
        try:
            # Dataframe creation
            df = pd.DataFrame(
                {
                "sepal length (cm)": [sepal_height], "sepal width (cm)": [sepal_width], 
                "petal length (cm)": [petal_height], "petal width (cm)": [petal_width],
                }
            )
            print(f"[Info] Input data as dataframe :\n{df.to_markdown()}")
            # df.columns = num_cols+cat_cols
            # df = df[num_cols+cat_cols] # reorder the dateframe

            # ML part 
            output = end2end_pipeline.predict_proba(df)
            
            ## store confidence score/ probability for the predicted class
            confidence_score = output.max(axis=-1)
            df['confidence score'] = confidence_score

            ## get index of the predicted class
            predicted_idx = output.argmax(axis=-1)
            
            # store index then replace by the matching label
            df['predicted label'] = predicted_idx
            predicted_label = df['predicted label'].replace(idx_to_labels)
            df['predicted label'] = predicted_label

            print(f"[Info] Input dataframe with prediction :\n{df.to_markdown()}")
            
            st.balloons()
            st.success(f"The iris has been classified as : '{predicted_label[0]}' with a confidence score of '{confidence_score[0]}' .", icon="✅")
            
        except:
            st.error(f"Something went wrong during the iris classification.", icon="🚨")