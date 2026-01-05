# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# st.markdown("""
# <style>
# /* Slider bar */
# div[data-baseweb="slider"] > div > div {
#     background: linear-gradient(90deg, #ff6f61, #ffa07a);
# }

# /* Slider handle */
# div[data-baseweb="slider"] span {
#     background-color: #ff6f61;
# }
# </style>
# """, unsafe_allow_html=True)
# # -------------------- SETUP --------------------
# st.set_page_config(page_title="Patient Data Manager", layout="wide")

# @st.cache_data
# def load_data():
#     return pd.read_csv("AED4weeks.csv")

# df_original = load_data()
# df = df_original.copy()

# # -------------------- SIDEBAR FILTER --------------------
# st.sidebar.header("🔎 Filter Data")

# filtered_df = df.copy()

# for col in df.columns:
#     if col == "ID":
#         continue

#     # Numeric columns → slider
#     if pd.api.types.is_numeric_dtype(df[col]):
#         min_val = int(df[col].min())
#         max_val = int(df[col].max())

#         selected_range = st.sidebar.slider(
#             f"{col}",
#             min_val,
#             max_val,
#             (min_val, max_val)
#         )

#         filtered_df = filtered_df[
#             (filtered_df[col] >= selected_range[0]) &
#             (filtered_df[col] <= selected_range[1])
#         ]

#     # Categorical columns → multiselect
#     else:
#         options = df[col].dropna().unique().tolist()
#         selected_vals = st.sidebar.multiselect(
#             f"{col}",
#             options,
#             default=options
#         )

#         filtered_df = filtered_df[
#             filtered_df[col].isin(selected_vals)
#         ]

# # -------------------- MAIN TABLE --------------------
# st.title("📋 Patient Data Management System")

# st.subheader("Filtered Dataset")
# st.dataframe(filtered_df, use_container_width=True)

# # -------------------- MODIFY / DELETE --------------------
# st.divider()
# st.subheader("✏️ Modify / 🗑️ Delete Patient Record")

# action = st.radio(
#     "Choose action:",
#     ["Modify", "Delete"],
#     horizontal=True
# )

# pid = st.text_input("Enter Patient ID")

# if pid:
#     record = filtered_df[filtered_df["ID"].astype(str) == pid]

#     if record.empty:
#         st.error("Patient ID not found.")
#     else:
#         st.success("Patient found:")
#         st.dataframe(record)

#         if action == "Modify":
#             col_to_modify = st.selectbox(
#                 "Select column to modify",
#                 [c for c in df.columns if c != "ID"]
#             )

#             new_value = st.text_input("Enter new value")

#             if st.button("Confirm Modify"):
#                 idx = df[df["ID"].astype(str) == pid].index[0]
#                 df.loc[idx, col_to_modify] = new_value
#                 st.success("Record updated successfully.")

#         elif action == "Delete":
#             if st.button("Confirm Delete"):
#                 idx = df[df["ID"].astype(str) == pid].index
#                 df.drop(idx, inplace=True)
#                 st.success("Record deleted successfully.")

# # -------------------- VISUALIZATION --------------------
# st.divider()
# st.subheader("📊 Data Visualization")

# col1, col2 = st.columns(2)

# # Bar chart: BreachOrNot
# st.subheader("📊 Breach Distribution (Ratio)")

# # chọn cột breach (không hard-code)
# cat_cols = [
#     c for c in df.columns
#     if not pd.api.types.is_numeric_dtype(df[c]) and c != "ID"
# ]

# breach_col = st.selectbox(
#     "Select breach indicator column",
#     cat_cols
# )

# breach_counts = filtered_df[breach_col].value_counts()

# fig, ax = plt.subplots()
# ax.pie(
#     breach_counts,
#     labels=breach_counts.index,
#     autopct="%1.1f%%",
#     startangle=90,
#     wedgeprops={"edgecolor": "white"}
# )
# ax.axis("equal")  # hình tròn đẹp

# st.pyplot(fig)


# # Additional chart example
# with col2:
#     st.markdown("### Numeric Variable Distribution")
#     num_cols = df.select_dtypes(include="number").columns.tolist()

#     selected_num = st.selectbox("Select numeric variable", num_cols)

#     fig2, ax2 = plt.subplots()
#     filtered_df[selected_num].hist(bins=20, ax=ax2)
#     ax2.set_xlabel(selected_num)
#     ax2.set_ylabel("Frequency")

#     st.pyplot(fig2)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Patient Data Manager", layout="wide")

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data():
    return pd.read_csv("AED4weeks.csv")

df_original = load_data()

# -------------------- SESSION STATE INIT --------------------
if "df" not in st.session_state:
    st.session_state["df"] = df_original.copy()

if "logs" not in st.session_state:
    st.session_state["logs"] = []

def reset_to_original():
    st.session_state["df"] = df_original.copy()
    st.session_state["logs"] = []
    st.success("Reset to original completed ✅")

df = st.session_state["df"]  # source of truth

# -------------------- HELPERS --------------------
def cast_value(series: pd.Series, raw: str):
    """Cast new value to column dtype when possible."""
    if raw is None:
        return raw
    raw = str(raw)

    if pd.api.types.is_numeric_dtype(series):
        try:
            if pd.api.types.is_integer_dtype(series):
                return int(float(raw))
            return float(raw)
        except:
            return raw
    return raw

# -------------------- SIDEBAR FILTER --------------------
st.sidebar.header("🔎 Filter Data")
st.sidebar.button("♻️ Reset to original", on_click=reset_to_original)

filtered_df = df.copy()

for col in df.columns:
    if col == "ID":
        continue

    # ---------- NUMERIC: DOUBLE-ENDED SLIDER (RANGE) ----------
    if pd.api.types.is_numeric_dtype(df[col]):
        s = df[col].dropna()
        if s.empty:
            continue

        cmin, cmax = s.min(), s.max()

        # IMPORTANT: value MUST be a tuple (min, max) => double-ended slider
        if pd.api.types.is_integer_dtype(df[col]):
            min_val = int(cmin)
            max_val = int(cmax)

            selected_range = st.sidebar.slider(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),   # <-- double-ended
                step=1,
                key=f"rng_{col}"
            )
        else:
            min_val = float(cmin)
            max_val = float(cmax)
            step = (max_val - min_val) / 100 if max_val != min_val else 0.1

            selected_range = st.sidebar.slider(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),   # <-- double-ended
                step=step,
                key=f"rng_{col}"
            )

        filtered_df = filtered_df[
            (filtered_df[col] >= selected_range[0]) &
            (filtered_df[col] <= selected_range[1])
        ]

    # ---------- CATEGORICAL: MULTISELECT ----------
    else:
        options = df[col].dropna().unique().tolist()
        selected_vals = st.sidebar.multiselect(
            label=col,
            options=options,
            default=options,
            key=f"ms_{col}"
        )
        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

# -------------------- MAIN TABLE --------------------
st.title("📋 Patient Data Management System")
st.subheader("Filtered Dataset")
st.dataframe(filtered_df, use_container_width=True)

# -------------------- MODIFY / DELETE --------------------
st.divider()
st.subheader("✏️ Modify / 🗑️ Delete Patient Record")

action = st.radio("Choose action:", ["Modify", "Delete"], horizontal=True)
pid = st.text_input("Enter Patient ID")

if pid:
    record = df[df["ID"].astype(str) == str(pid)]  # search in full df

    if record.empty:
        st.error("Patient ID not found in the main dataset.")
    else:
        st.success("Patient found (current row in main dataset):")
        st.dataframe(record, use_container_width=True)

        if action == "Modify":
            col_to_modify = st.selectbox(
                "Select column to modify",
                [c for c in df.columns if c != "ID"]
            )
            new_value_raw = st.text_input("Enter new value")

            if st.button("Confirm Modify"):
                idx = df.index[df["ID"].astype(str) == str(pid)].tolist()[0]

                before_row = df.loc[[idx]].copy()
                old_val = df.loc[idx, col_to_modify]

                new_val = cast_value(df[col_to_modify], new_value_raw)
                st.session_state["df"].loc[idx, col_to_modify] = new_val

                after_row = st.session_state["df"].loc[[idx]].copy()

                st.session_state["logs"].append({
                    "action": "MODIFY",
                    "ID": str(pid),
                    "column": col_to_modify,
                    "old_value": str(old_val),
                    "new_value": str(new_val),
                    "before_row": before_row.to_dict(orient="records")[0],
                    "after_row": after_row.to_dict(orient="records")[0],
                })

                st.success("Record updated successfully ✅")

        elif action == "Delete":
            if st.button("Confirm Delete"):
                idx_list = df.index[df["ID"].astype(str) == str(pid)].tolist()

                before_rows = df.loc[idx_list].copy()
                st.session_state["df"].drop(index=idx_list, inplace=True)

                st.session_state["logs"].append({
                    "action": "DELETE",
                    "ID": str(pid),
                    "column": None,
                    "old_value": None,
                    "new_value": None,
                    "before_row": before_rows.to_dict(orient="records"),
                    "after_row": None,
                })

                st.success("Record deleted successfully ✅")

# -------------------- CHANGE LOG --------------------
st.divider()
st.subheader("🧾 Change Log (Before vs After)")

if len(st.session_state["logs"]) == 0:
    st.info("No changes yet.")
else:
    log_df = pd.DataFrame(st.session_state["logs"])
    st.dataframe(log_df[["action", "ID", "column", "old_value", "new_value"]], use_container_width=True)

    pick = st.number_input(
        "Log index",
        min_value=0,
        max_value=len(st.session_state["logs"]) - 1,
        value=len(st.session_state["logs"]) - 1,
        step=1
    )

    st.write("**Before:**")
    st.json(st.session_state["logs"][pick]["before_row"])
    st.write("**After:**")
    st.json(st.session_state["logs"][pick]["after_row"])

# -------------------- VISUALIZATION --------------------
st.divider()
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Breach Distribution (Ratio)")
    cat_cols = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])) and c != "ID"]

    if not cat_cols:
        st.warning("No categorical columns found.")
    else:
        breach_col = st.selectbox("Select breach indicator column", cat_cols)
        breach_counts = filtered_df[breach_col].value_counts(dropna=False)

        fig, ax = plt.subplots()
        ax.pie(
            breach_counts,
            labels=breach_counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white"}
        )
        ax.axis("equal")
        st.pyplot(fig)

with col2:
    st.subheader("Numeric Variable Distribution")
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        st.warning("No numeric columns found.")
    else:
        selected_num = st.selectbox("Select numeric variable", num_cols)
        fig2, ax2 = plt.subplots()
        ax2.hist(filtered_df[selected_num].dropna(), bins=20)
        ax2.set_xlabel(selected_num)
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
