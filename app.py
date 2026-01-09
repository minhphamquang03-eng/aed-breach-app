import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pulp
import numpy as np
import json, hashlib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Patient Data Manager", layout="wide")

st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose module", ["Optimization","AED"], index=0)

if app_mode == "AED":
    # -------------------- DATA LOADING --------------------
    @st.cache_data
    def load_data():
        return pd.read_csv("AED4weeks.csv.gz", compression="gzip")

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

    # -------------------- FORCE CATEGORICAL FOR VIZ/SUMMARY --------------------
    FORCE_CAT_COLS = [c for c in ["Day", "Period"] if c in df.columns]

    def get_num_cols_for_viz(dframe: pd.DataFrame):
        return [c for c in dframe.select_dtypes(include="number").columns
                if c not in (["ID"] + FORCE_CAT_COLS)]

    def get_cat_cols_for_viz(dframe: pd.DataFrame):
        base = [c for c in dframe.columns
                if c != "ID" and (not pd.api.types.is_numeric_dtype(dframe[c]))]
        for c in FORCE_CAT_COLS:
            if c in dframe.columns and c not in base:
                base.append(c)
        return base

    # -------------------- SAMPLING --------------------
    st.sidebar.divider()
    st.sidebar.subheader("🎲 Sampling (Optional)")

    use_sample = st.sidebar.checkbox("Use sample", value=False)

    max_n = len(df)
    default_n = min(400, max_n)

    sample_n = st.sidebar.number_input(
        "Sample size (n)",
        min_value=10,
        max_value=max_n,
        value=default_n,
        step=10
    )

    sample_seed = st.sidebar.number_input(
        "Seed",
        min_value=0,
        max_value=10_000,
        value=42,
        step=1
    )

    if use_sample:
        df_work = df.sample(n=int(sample_n), random_state=int(sample_seed)).copy()
        st.sidebar.success(f"Using SAMPLE: n={int(sample_n)}, seed={int(sample_seed)}")
    else:
        df_work = df.copy()
        st.sidebar.info("Using FULL dataset")

    # -------------------- HELPERS --------------------
    def cast_value(series: pd.Series, raw: str):
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
    st.sidebar.button("♻️ Reset to original", on_click=reset_to_original)
    st.sidebar.header("🔎 Filter Data")

    filtered_df = df_work.copy()

    # ----------- ID FILTER -----------
    st.sidebar.subheader("🆔 Filter by Patient ID")
    id_input = st.sidebar.text_input(
        "Enter patient ID(s) (comma-separated)",
        placeholder="e.g. P10510, P10511"
    )
    if id_input.strip():
        id_list = [i.strip() for i in id_input.split(",") if i.strip() != ""]
        filtered_df = filtered_df[filtered_df["ID"].astype(str).isin(id_list)]

    # ----------- Column filters -----------
    for col in df_work.columns:
        if col == "ID":
            continue

        # ✅ Day/Period vẫn slider
        use_slider = pd.api.types.is_numeric_dtype(df_work[col]) or (col in FORCE_CAT_COLS)

        if use_slider:
            s = df_work[col].dropna()
            if s.empty:
                continue

            cmin, cmax = s.min(), s.max()

            if pd.api.types.is_integer_dtype(df_work[col]) or (col in FORCE_CAT_COLS):
                min_val, max_val = int(cmin), int(cmax)
                selected_range = st.sidebar.slider(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1,
                    key=f"rng_{col}"
                )
            else:
                min_val, max_val = float(cmin), float(cmax)
                step = (max_val - min_val) / 100 if max_val != min_val else 0.1
                selected_range = st.sidebar.slider(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step,
                    key=f"rng_{col}"
                )

            filtered_df = filtered_df[
                (filtered_df[col] >= selected_range[0]) &
                (filtered_df[col] <= selected_range[1])
            ]
        else:
            options = df_work[col].dropna().astype(str).unique().tolist()
            selected_vals = st.sidebar.multiselect(
                label=col,
                options=options,
                default=options,
                key=f"ms_{col}"
            )
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_vals)]

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
        record = df[df["ID"].astype(str) == str(pid)]

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
                    st.rerun()

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
                    st.rerun()

    # -------------------- CHANGE LOG --------------------
    st.divider()
    st.subheader("🧾 Change Log (Before vs After)")

    if len(st.session_state["logs"]) == 0:
        st.info("No changes yet.")
    else:
        log_df = pd.DataFrame(st.session_state["logs"])
        st.dataframe(log_df[["action", "ID", "column", "old_value", "new_value"]], use_container_width=True)

    # -------------------- SUMMARY STATISTICS --------------------
    st.divider()
    st.subheader("📊 Summary Statistics")

    df_sum = filtered_df.copy()

    # ✅ df_viz: ép Day/Period thành string để summary/chart coi là categorical
    df_viz = df_sum.copy()
    for c in FORCE_CAT_COLS:
        df_viz[c] = df_viz[c].astype(str)

    # ---- Numerical summary (exclude Day/Period) ----
    num_cols_viz = get_num_cols_for_viz(df_viz)
    st.markdown("#### 🔢 Numerical Variables")
    if num_cols_viz:
        num_summary = (
            df_viz[num_cols_viz]
            .describe()
            .T
            .round(4)
            .reset_index()
            .rename(columns={"index": "Variable"})
        )
        st.dataframe(num_summary, use_container_width=True)
    else:
        st.info("No numerical variables found (excluding Day/Period).")

    # # ---- Categorical summary (count + share) ----
    # st.markdown("### 🏷️ Categorical Variables (Count & Share)")
    cat_cols_viz = get_cat_cols_for_viz(df_viz)
    # if cat_cols_viz:
    #     cat_pick = st.selectbox("Select categorical variable for summary", cat_cols_viz, key="cat_sum")
    #     counts = df_viz[cat_pick].value_counts(dropna=False)
    #     share = (counts / counts.sum() * 100).round(2)
    #     cat_table = pd.DataFrame({"Count": counts, "Share (%)": share})
    #     st.dataframe(cat_table, use_container_width=True)
    # else:
    #     st.info("No categorical variables found.")

    # -------------------- VISUALIZATION --------------------

    # (1) TOP: Numeric distribution FULL WIDTH
    st.markdown("#### Numeric Variable Distribution ")
    
    plot_type = st.radio(
        "Choose plot type",
        ["Histogram/Bar", "Boxplot"],
        horizontal=True,
        key="num_dist_plot_type"
    )

    if not num_cols_viz:
        st.warning("No numeric columns found (excluding Day/Period).")
    else:
        selected_num = st.selectbox("Select numeric variable", num_cols_viz, key="num_fullwidth")
        x = df_viz[selected_num].dropna()

        if plot_type == "Boxplot":
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.boxplot(x.values, vert=False)
            ax.set_xlabel(selected_num)
            ax.set_title(f"Boxplot of {selected_num}")
            st.pyplot(fig)

        else:  # Histogram/Bar
            if pd.api.types.is_integer_dtype(x) and x.nunique() <= 50:
                counts = x.value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(counts.index.astype(int), counts.values, width=0.8, color="#AB2346")
                ax.set_xlabel(selected_num)
                ax.set_ylabel("Frequency")
                ax.set_xticks(counts.index.astype(int))
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(x, bins=20, color="#AB2346")
                ax.set_xlabel(selected_num)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)


    st.divider()

    # (2) BELOW: Left chart + Right table
    st.markdown('#### 🏷️ Categorical Variables')
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Categorical Distribution (Count)")
        if not cat_cols_viz:
            st.warning("No categorical columns found.")
        else:
            cat_col = st.selectbox("Select categorical variable", cat_cols_viz, key="cat_bar")
            counts = df_viz[cat_col].value_counts(dropna=False)

            counts_df = counts.rename_axis(cat_col).reset_index(name="Count")

            counts_df["_sort_key"] = pd.to_numeric(counts_df[cat_col].astype(str), errors="coerce")

            counts_df = counts_df.sort_values("_sort_key", na_position="last")
            counts = counts_df.set_index(cat_col)["Count"]



            figC, axC = plt.subplots(figsize=(8, 4))
            axC.bar(counts.index.astype(str), counts.values, color = "#237944")
            axC.set_title(f"Distribution of {cat_col}")
            axC.set_xlabel(cat_col)
            axC.set_ylabel("Number of patients")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(figC)

    
    with col_right:
        st.markdown("#### Table: Count & Share (%)")
        if not cat_cols_viz:
            st.info("No categorical variables.")
        else:
        
            share = (counts / counts.sum() * 100).round(2)
            st.dataframe(pd.DataFrame({"Count": counts, "Share (%)": share}), use_container_width=True)

    # -------------------- BOXPLOT: NUMERIC vs BREACH --------------------
    st.divider()


    col_left1, col_right1 = st.columns([5, 6])

    with col_left1:
        st.markdown("#### 📦 Numeric vs Breach")
        breach_col_name = "Breachornot"

        if breach_col_name not in df_viz.columns:
            st.warning("Breachornot column not found.")
        else:
            if not num_cols_viz:
                st.warning("No numeric variables available for boxplot (excluding Day/Period).")
            else:
                num_var = st.selectbox("Select numeric variable", num_cols_viz, key="boxplot_num")

                plot_df = df_viz[[num_var, breach_col_name]].dropna().copy()
                if plot_df.empty:
                    st.warning("No data available after filtering.")
                else:
                    # --- make breach labels stable across environments ---
                    # Handles 0/1, "0"/"1", "breach/non-breach", "Breach/Non-breach", etc.
                    s = plot_df[breach_col_name]

                    if pd.api.types.is_numeric_dtype(s):
                        plot_df["Breach_label"] = s.astype(int).map({0: "non-breach", 1: "breach"})
                    else:
                        ss = s.astype(str).str.strip().str.lower()
                        mapping = {
                            "0": "non-breach", "no": "non-breach", "n": "non-breach",
                            "non-breach": "non-breach", "nonbreach": "non-breach", "non breached": "non-breach",
                            "1": "breach", "yes": "breach", "y": "breach",
                            "breach": "breach", "breached": "breach"
                        }
                        plot_df["Breach_label"] = ss.map(mapping).fillna(ss)

                    # keep only the two main groups (optional, but cleaner)
                    plot_df = plot_df[plot_df["Breach_label"].isin(["non-breach", "breach"])]

                    if plot_df.empty:
                        st.warning("Breach labels could not be parsed into breach/non-breach.")
                    else:
                        # --- fixed order + fixed colors (stable on Streamlit Cloud) ---
                        order = ["non-breach", "breach"]
                        palette = {"non-breach": "#4C72B0", "breach": "#DD8452"}

                        figB, axB = plt.subplots(figsize=(8, 4))
                        sns.boxplot(
                            data=plot_df,
                            x="Breach_label",
                            y=num_var,
                            order=order,
                            palette=palette,
                            ax=axB
                        )
                        axB.set_title(f"{num_var} by Breach Status")
                        axB.set_xlabel("Breach status")
                        axB.set_ylabel(num_var)
                        st.pyplot(figB)

    with col_right1:
        st.markdown("#### 📊 Categorical vs Breach ")

        # ----- choose breach column (auto detect) -----
        breach_candidates = [c for c in df.columns if c.lower() in ["breachornot", "breach", "breached"]]
        breach_col = breach_candidates[0] if breach_candidates else None

        if breach_col is None:
            st.warning("No breach column found (expected 'Breachornot').")
        else:
            # cat columns for this chart (Day/Period treated as categorical too)
            df_stack = filtered_df.copy()
            for c in FORCE_CAT_COLS:
                if c in df_stack.columns:
                    df_stack[c] = df_stack[c].astype(str)

            cat_cols_stack = get_cat_cols_for_viz(df_stack)

            if not cat_cols_stack:
                st.warning("No categorical columns found.")
            else:
                cat_pick = st.selectbox("Select categorical variable", cat_cols_stack, key="stack_cat")

                show_percent = st.checkbox("Show as % (normalized)", value=True, key="stack_pct")

                # crosstab
                ct = pd.crosstab(
                    df_stack[cat_pick].astype(str),
                    df_stack[breach_col].astype(str)
                )

                # (optional) đảm bảo có đủ 2 cột để plot ổn định
                for k in ["breach", "non-breach"]:
                    if k not in ct.columns:
                        ct[k] = 0
                ct = ct[["breach", "non-breach"]]  # ✅ breach trước

                # sort by numeric order if possible (Day/Period numeric-like)
                idx_num = pd.to_numeric(ct.index, errors="coerce")
                if idx_num.notna().all():
                    ct = ct.iloc[idx_num.argsort()]  # ✅ đúng, không dùng .index

                # normalize
                if show_percent:
                    plot_data = ct.div(ct.sum(axis=1), axis=0) * 100
                    ylabel = "Percentage (%)"
                else:
                    plot_data = ct
                    ylabel = "Count"

                # plot (✅ breach ở dưới để nằm "trên" trong stack? -> muốn breach nằm trên thì breach phải là layer thứ 2)
                fig, ax = plt.subplots(figsize=(10, 4))

                ax.bar(
                    plot_data.index.astype(str),
                    plot_data["breach"],
                    label="breach",
                    color = '#3B6F91'
                )

                ax.bar(
                    plot_data.index.astype(str),
                    plot_data["non-breach"],
                    bottom=plot_data["breach"],
                    label="non-breach",
                    color = '#ED7D3A'
                )

                ax.set_ylabel(ylabel)
                ax.set_xlabel(cat_pick)
                ax.set_title(f"{cat_pick} vs {breach_col} ({'%' if show_percent else 'Count'})")

                # ✅ legend xuống dưới
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.8, -0.25),
                    ncol=2,
                    frameon=True
                )

                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

    # =========================
    # 🔁 Flexible 2-variable Explorer
    # Rules:
    #   numeric vs numeric      -> median line plot (+ IQR band)
    #   numeric vs categorical  -> boxplot
    #   categorical vs categorical -> heatmap (crosstab)
    # =========================

    st.divider()
    st.subheader("🧪 Explore relationship between any 2 variables (except Breachornot)")

    BREACH_COL = "Breachornot"
    FORCE_CAT_COLS = [c for c in ["Day", "Period"] if c in df_viz.columns]  # df_viz is your filtered_df copy for viz

    # Candidate columns (exclude ID & breach)
    cand_cols = [c for c in df_viz.columns if c not in ["ID", BREACH_COL]]

    if len(cand_cols) < 2:
        st.warning("Not enough columns to compare.")
    else:
        colA, colB = st.columns(2)
        with colA:
            var_x = st.selectbox("Select variable X", cand_cols, index=0, key="rel_x")
        with colB:
            var_y = st.selectbox("Select variable Y", cand_cols, index=1, key="rel_y")

        # --- helper: decide if a column should be treated as categorical for plotting ---
        def is_cat_for_plot(dframe: pd.DataFrame, colname: str) -> bool:
            if colname in FORCE_CAT_COLS:
                return True
            return (not pd.api.types.is_numeric_dtype(dframe[colname]))

        x_is_cat = is_cat_for_plot(df_viz, var_x)
        y_is_cat = is_cat_for_plot(df_viz, var_y)

        # Data for plot (drop rows where either is missing)
        plot_df = df_viz[[var_x, var_y]].dropna()
        if plot_df.empty:
            st.warning("No data available after filtering (all rows missing in selected variables).")
        else:
            # =========================
        # 1) numeric vs numeric -> median line plot (+ IQR band)
        #    FIX: if X is discrete integer with small #unique -> group by actual X values
        # =========================
            if (not x_is_cat) and (not y_is_cat):
                st.markdown("#### 📈 Numeric vs Numeric: Median line plot (Y by X)")

                x = pd.to_numeric(plot_df[var_x], errors="coerce")
                y = pd.to_numeric(plot_df[var_y], errors="coerce")
                tmp = pd.DataFrame({var_x: x, var_y: y}).dropna()

                if tmp.empty:
                    st.warning("Selected numeric variables cannot be converted to numeric for plotting.")
                else:
                    x_unique = tmp[var_x].nunique(dropna=True)

                    # Heuristic: discrete integer-like and not too many unique values -> group by value
                    is_int_like = (tmp[var_x] % 1 == 0).all()
                    if is_int_like and x_unique <= 30:
                        g = tmp.groupby(var_x)[var_y]
                        med = g.median()
                        q1 = g.quantile(0.25)
                        q3 = g.quantile(0.75)

                        xs = med.index.values  # actual X values (e.g., 0..6)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(xs, med.values, marker="o", linewidth=2, label="Median")
                        ax.fill_between(xs, q1.values, q3.values, alpha=0.25, label="IQR (25–75%)")
                        ax.set_xlabel(var_x)
                        ax.set_ylabel(var_y)
                        ax.set_title(f"Median {var_y} across {var_x} (by value)")
                        ax.set_xticks(sorted(xs))
                        ax.legend(frameon=False)
                        st.pyplot(fig)

                    else:
                        # Continuous X -> keep binned quantile trend
                        bins = st.slider("Number of bins for X (for median trend)", 5, 30, 10, 1, key="nn_bins")
                        try:
                            tmp["_xbin"] = pd.qcut(tmp[var_x], q=bins, duplicates="drop")
                        except Exception:
                            tmp["_xbin"] = pd.cut(tmp[var_x], bins=bins)

                        g = tmp.groupby("_xbin")[var_y]
                        med = g.median()
                        q1 = g.quantile(0.25)
                        q3 = g.quantile(0.75)

                        mids = [b.mid for b in med.index]
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(mids, med.values, marker="o", linewidth=2, label="Median")
                        ax.fill_between(mids, q1.values, q3.values, alpha=0.25, label="IQR (25–75%)")
                        ax.set_xlabel(var_x)
                        ax.set_ylabel(var_y)
                        ax.set_title(f"Median {var_y} across {var_x} (binned)")
                        ax.legend(frameon=False)
                        st.pyplot(fig)


            # =========================
            # 2) numeric vs categorical -> boxplot
            # =========================
            elif x_is_cat ^ y_is_cat:
                st.markdown("#### 📦 Numeric vs Categorical: Boxplot")

                # make categorical on x-axis, numeric on y-axis
                if x_is_cat and (not y_is_cat):
                    cat_col, num_col = var_x, var_y
                else:
                    cat_col, num_col = var_y, var_x

                # Convert forced-cat to string for stable ordering display
                tmp = plot_df.copy()
                tmp[cat_col] = tmp[cat_col].astype(str)

                # Optional: if category looks numeric (Day/Period), order by numeric
                cat_order = tmp[cat_col].unique().tolist()
                as_num = pd.to_numeric(pd.Series(cat_order), errors="coerce")
                if as_num.notna().all():
                    cat_order = [x for _, x in sorted(zip(as_num.tolist(), cat_order))]

                fig, ax = plt.subplots(figsize=(10, 4))

                sns.boxplot(
                    data=tmp,
                    x=cat_col,
                    y=num_col,
                    order=cat_order,
                    ax=ax,
                    color="#4C72B0",        # ✅ ép 1 màu duy nhất
                    linewidth=1.2,
                    fliersize=3
                )

                ax.set_title(f"{num_col} by {cat_col}")
                ax.set_xlabel(cat_col)
                ax.set_ylabel(num_col)
                plt.xticks(rotation=45, ha="right")

                st.pyplot(fig)


            # =========================
            # 3) categorical vs categorical -> heatmap (crosstab)
            # =========================
            else:
                st.markdown("#### 🔥 Categorical vs Categorical: Heatmap (Crosstab)")

                tmp = plot_df.copy()
                tmp[var_x] = tmp[var_x].astype(str)
                tmp[var_y] = tmp[var_y].astype(str)

                ct = pd.crosstab(tmp[var_x], tmp[var_y])

                # optional: show % by row
                show_pct = st.checkbox("Show as row % (normalized)", value=False, key="cc_pct")
                data = (ct.div(ct.sum(axis=1), axis=0) * 100) if show_pct else ct

                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(data, annot=False, cmap="Blues", ax=ax)
                ax.set_title(f"{var_x} vs {var_y}" + (" (Row %)" if show_pct else " (Count)"))
                ax.set_xlabel(var_y)
                ax.set_ylabel(var_x)
                st.pyplot(fig)

                # show table too (useful)
                st.dataframe(data.round(2) if show_pct else data, use_container_width=True)


                # # optional: show table
                # st.markdown("**Crosstab table**")
                # if show_percent:
                #     st.dataframe(plot_data.round(2), use_container_width=True)
                # else:
                #     st.dataframe(plot_data, use_container_width=True)

    # # -------------------- Stacked bar chart: categorical vs BREACH --------------------
    # st.divider()
    # st.markdown("### 📊 Categorical vs Breach (Stacked Bar)")

    # # ----- choose breach column (auto detect) -----
    # breach_candidates = [c for c in df.columns if c.lower() in ["breachornot", "breach", "breached"]]
    # breach_col = breach_candidates[0] if breach_candidates else None

    # if breach_col is None:
    #     st.warning("No breach column found (expected 'Breachornot').")
    # else:
    #     # cat columns for this chart (Day/Period treated as categorical too)
    #     df_stack = filtered_df.copy()
    #     for c in FORCE_CAT_COLS:
    #         if c in df_stack.columns:
    #             df_stack[c] = df_stack[c].astype(str)

    #     cat_cols_stack = get_cat_cols_for_viz(df_stack)

    #     if not cat_cols_stack:
    #         st.warning("No categorical columns found.")
    #     else:
    #         cat_pick = st.selectbox("Select categorical variable", cat_cols_stack, key="stack_cat")

    #         show_percent = st.checkbox("Show as % (normalized)", value=True, key="stack_pct")

    #         # crosstab
    #         ct = pd.crosstab(df_stack[cat_pick].astype(str), df_stack[breach_col].astype(str))

    #         # sort by numeric order if possible (Day/Period are numeric-like)
    #         idx_num = pd.to_numeric(ct.index, errors="coerce")
    #         if idx_num.notna().all():
    #             ct = ct.iloc[idx_num.argsort()]

    #         # normalize to percent if needed
    #         plot_data = ct.div(ct.sum(axis=1), axis=0) * 100 if show_percent else ct

    #         # plot
    #         fig, ax = plt.subplots(figsize=(8, 4))
    #         plot_data.plot(kind="bar", stacked=True, ax=ax)

    #         ax.set_title(f"{cat_pick} vs {breach_col} ({'%' if show_percent else 'Count'})")
    #         ax.set_xlabel(cat_pick)
    #         ax.set_ylabel("Percentage (%)" if show_percent else "Count")
    #         ax.legend(title=breach_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    #         plt.xticks(rotation=45, ha="right")
    #         plt.tight_layout()

    #         st.pyplot(fig)

    #         # optional: show table
    #         st.markdown("**Crosstab table**")
    #         if show_percent:
    #             st.dataframe(plot_data.round(2), use_container_width=True)
    #         else:
    #             st.dataframe(plot_data, use_container_width=True)


else:
    st.title("📈 Optimization Module")


    # -------------------------
    # 0) Choose objective FIRST
    # -------------------------
    opt_mode = st.sidebar.selectbox(
        "Choose optimization objective",
        ["Minimize cost", "Balance workload (fairness)"],
        index=0
    )

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Constraints")

    # -------------------------
    # 1) Base data (bạn chỉnh theo notebook của bạn)
    # -------------------------
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    ops = ["E. Khan", "Y. Chen", "A. Taylor", "R. Zidane", "R. Perez", "C. Santos"]  # đổi tên theo “students/operators” của bạn
    skills = ["Programming", "Troubleshooting"]

    DEFAULT_SKILL_MAP = {
        ("E. Khan","Troubleshooting"): True, ("E. Khan","Programming"): False,
        ("Y. Chen","Troubleshooting"): True, ("Y. Chen","Programming"): False,
        ("A. Taylor","Troubleshooting"): False, ("A. Taylor","Programming"): True,
        ("R. Zidane","Troubleshooting"): False, ("R. Zidane","Programming"): True,
        ("R. Perez","Troubleshooting"): True, ("R. Perez","Programming"): False,
        ("C. Santos","Troubleshooting"): True, ("C. Santos","Programming"): True,
    }

    # wage per operator 
    wage = {"E. Khan":25, "Y. Chen":26, "A. Taylor":24, "R. Zidane":23, "R. Perez":28, "C. Santos":30}

    # min weekly hours per operator 
    min_week_default = {"E. Khan":8, "Y. Chen":8, "A. Taylor":8, "R. Zidane":8, "R. Perez":7, "C. Santos":7}

    # availability of each operators
    DEFAULT_AVAIL =  {
    ("E. Khan","Mon"):6, ("E. Khan","Tue"):0, ("E. Khan","Wed"):6, ("E. Khan","Thu"):0, ("E. Khan","Fri"):6,
    ("Y. Chen","Mon"):0, ("Y. Chen","Tue"):6, ("Y. Chen","Wed"):0, ("Y. Chen","Thu"):6, ("Y. Chen","Fri"):0,
    ("A. Taylor","Mon"):4, ("A. Taylor","Tue"):8, ("A. Taylor","Wed"):4, ("A. Taylor","Thu"):0, ("A. Taylor","Fri"):4,
    ("R. Zidane","Mon"):5, ("R. Zidane","Tue"):5, ("R. Zidane","Wed"):5, ("R. Zidane","Thu"):0, ("R. Zidane","Fri"):5,
    ("R. Perez","Mon"):3, ("R. Perez","Tue"):0, ("R. Perez","Wed"):3, ("R. Perez","Thu"):8, ("R. Perez","Fri"):0,
    ("C. Santos","Mon"):0, ("C. Santos","Tue"):0, ("C. Santos","Wed"):0, ("C. Santos","Thu"):6, ("C. Santos","Fri"):2
}


    def skill_map_to_nested(skill_map: dict):
        # (op, skill) -> v  ==>  {op: {skill: v}}
        out = {}
        for (op, sk), v in skill_map.items():
            op = str(op); sk = str(sk)
            out.setdefault(op, {})[sk] = int(v)
        return out

    def make_settings_key(avail_df: pd.DataFrame, other_params: dict):
        # avail_df: stable json-safe
        avail_payload = {
            "index": [str(i) for i in avail_df.index.tolist()],
            "columns": [str(c) for c in avail_df.columns.tolist()],
            "data": avail_df.astype(float).fillna(0).values.tolist(),
        }

        # other_params: make json-safe (tuple-key dicts -> nested dict)
        safe_params = {}
        for k, v in other_params.items():
            if k == "skill_map" and isinstance(v, dict):
                safe_params[k] = skill_map_to_nested(v)
            else:
                safe_params[k] = v

        payload = {"avail": avail_payload, "params": safe_params}
        s = json.dumps(payload, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()


    # -------------------------
    # 2) Sidebar controls (constraints)
    # -------------------------
    # ========= Init availability table in session_state =========
    if "avail_df" not in st.session_state:
        init = pd.DataFrame(index=ops, columns=days, dtype=int)
        for o in ops:
            for d in days:
                init.loc[o, d] = int(DEFAULT_AVAIL.get((o, d), 0))
        st.session_state["avail_df"] = init

    # ========= Sidebar editor =========
    st.sidebar.text("Availability (hours per operator per day)")

    # nút reset nhanh
    if st.sidebar.button("Reset availability to default"):
        init = pd.DataFrame(index=ops, columns=days, dtype=int)
        for o in ops:
            for d in days:
                init.loc[o, d] = int(DEFAULT_AVAIL.get((o, d), 0))
        st.session_state["avail_df"] = init

    edited_avail_df = st.sidebar.data_editor(
        st.session_state["avail_df"],
        use_container_width=True,
        num_rows="fixed",
        key="avail_editor"
    )

    # ========= Clean / validate values (ensure int >= 0) =========
    edited_avail_df = edited_avail_df.copy()

    # clean columns
    edited_avail_df.columns = [str(c).strip() for c in edited_avail_df.columns]

    # if index got lost -> force back to ops
    edited_avail_df.index = edited_avail_df.index.astype(str)

    # IMPORTANT: ensure EXACT shape ops x days
    edited_avail_df = edited_avail_df.reindex(index=ops, columns=days).fillna(0)

    # numeric clean
    edited_avail_df = edited_avail_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    edited_avail_df = edited_avail_df.clip(lower=0).round(0).astype(int)




    # lưu lại
    st.session_state["avail_df"] = edited_avail_df

    # ví dụ: st.session_state["avail_df"] đã có rồi
    # =========================
    # SETTINGS KEY (FIX)
    # =========================

    # ========= Convert to avail dict (this is what constraints will use) =========
    avail = {(o, d): int(edited_avail_df.loc[o, d]) for o in ops for d in days}
    coverage_per_day = st.sidebar.slider(
        "Daily coverage hours (each weekday)",
        min_value=6, max_value=20, value=14, step=1
    )


    with st.sidebar.expander("Min weekly hours per operator", expanded=False):
        min_week = {}
        for o in ops:
            min_week[o] = st.number_input(
                f"Min weekly hours - {o}",
                min_value=0, max_value=80,
                value=int(min_week_default.get(o, 0)),
                step=1,
                key=f"minweek_{o}"
            )

    # (Optional) allow user adjust wages
    with st.sidebar.expander("Wage per operator (optional)", expanded=False):
        wage_ui = {}
        for o in ops:
            wage_ui[o] = st.number_input(
                f"Wage - {o}",
                min_value=0.0, max_value=200.0,
                value=float(wage.get(o, 0)),
                step=0.5,
                key=f"wage_{o}"
            )
    
    st.sidebar.markdown("### Skill availability")

    use_skill = st.sidebar.checkbox("Enable skill constraints", value=False)

    if "skill_df" not in st.session_state:
        init_skill = pd.DataFrame({"Operator": ops})
        for sk in skills:
            init_skill[sk] = [bool(DEFAULT_SKILL_MAP.get((o, sk), False)) for o in ops]
        st.session_state["skill_df"] = init_skill

    skill_req = 0.0
    skill_df_clean = st.session_state["skill_df"]

    if use_skill:
        st.sidebar.caption("Tick skills for each operator:")

        # table with checkboxes
        edited_skill_df = st.sidebar.data_editor(
            st.session_state["skill_df"],
            use_container_width=True,
            num_rows="fixed",
            key="skill_editor",
            disabled=["Operator"]  # lock name column
        )

        # clean types
        edited_skill_df = edited_skill_df.copy()
        edited_skill_df["Operator"] = edited_skill_df["Operator"].astype(str)
        for sk in skills:
            edited_skill_df[sk] = edited_skill_df[sk].astype(bool)

        st.session_state["skill_df"] = edited_skill_df
        skill_df_clean = edited_skill_df

        # single slider for requirement (applies to each selected skill)
        skill_req = st.sidebar.slider(
            "Required skilled hours per day (for each skill)",
            min_value=0.0,
            max_value=float(coverage_per_day),  # cannot exceed daily total coverage (nice bound)
            value=min(6.0, float(coverage_per_day)),
            step=1.0
        )
    
    def build_skill_map(skill_df: pd.DataFrame, skills: list):
        m = {}
        df2 = skill_df.set_index("Operator")
        for op in df2.index.astype(str):
            for sk in skills:
                m[(op, sk)] = 1 if bool(df2.loc[op, sk]) else 0
        return m

    def build_avail_dict_from_wide(avail_df: pd.DataFrame, ops: list, days: list):
        tmp = avail_df.copy()
        # ensure proper index/columns
        tmp.index = tmp.index.astype(str)
        tmp.columns = [c.strip() for c in tmp.columns]
        # numeric clean
        tmp = tmp.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
        avail = {(o, d): float(tmp.loc[o, d]) for o in ops for d in days}
        return avail

    def check_skill_feasibility(avail_dict: dict, skill_map: dict, skills: list, ops: list, days: list, skill_req: float):
        """
        Return a DataFrame with total available skilled hours per day/skill,
        and rows where total < skill_req (infeasible).
        """
        rows = []
        for d in days:
            for sk in skills:
                total = sum(avail_dict.get((o, d), 0.0) * skill_map.get((o, sk), 0) for o in ops)
                rows.append({
                    "Day": d,
                    "Skill": sk,
                    "Total_available_skilled_hours": round(total, 2),
                    "Required": float(skill_req),
                    "Gap": round(float(skill_req) - total, 2)
                })
        df_check = pd.DataFrame(rows)
        df_check["Feasible?"] = df_check["Total_available_skilled_hours"] >= df_check["Required"]
        return df_check

    # ------------------- run feasibility check (only if enabled) -------------------
    if use_skill and skill_req > 0:
        skill_map_ui = build_skill_map(st.session_state["skill_df"], skills)
        avail_dict_ui = build_avail_dict_from_wide(st.session_state["avail_df"], ops, days)

        df_skill_check = check_skill_feasibility(
            avail_dict=avail_dict_ui,
            skill_map=skill_map_ui,
            skills=skills,
            ops=ops,
            days=days,
            skill_req=skill_req
        )

        infeasible = df_skill_check[~df_skill_check["Feasible?"]].copy()

        # show warning + table (sidebar or main)
        if not infeasible.empty:
            st.sidebar.warning("⚠️ Skill requirement may be infeasible (some day/skill cannot reach required hours).")
            st.sidebar.dataframe(
                infeasible[["Day", "Skill", "Total_available_skilled_hours", "Required", "Gap"]],
                use_container_width=True
            )
        else:
            st.sidebar.success("✅ Skill constraints look feasible (based on availability).")


    
   #1) gom tất cả input ảnh hưởng nghiệm vào 1 dict
    # skill_map_ui must exist even if use_skill = False
    skill_map_ui = build_skill_map(st.session_state["skill_df"], skills)

    other_params = {
        "opt_mode": str(opt_mode),
        "coverage_per_day": int(coverage_per_day),
        "min_week": {o: int(min_week[o]) for o in ops},
        "wage_ui": {o: float(wage_ui[o]) for o in ops},
        "use_skill": bool(use_skill),
        "skill_req": float(skill_req),
        "skill_map": {(op, sk): int(val) for (op, sk), val in skill_map_ui.items()},  # OK vì make_settings_key sẽ convert
    }



    # 2) key sẽ thay đổi nếu bất kỳ thứ gì ở trên đổi
    settings_key = make_settings_key(st.session_state["avail_df"], other_params)

    # 3) solve lại nếu key khác hoặc chưa có kết quả lần nào
    need_solve = (st.session_state.get("last_settings_key") != settings_key) or ("opt_result" not in st.session_state)
    
    # import time
    # auto_run = st.sidebar.checkbox("⚡ Auto-run optimization", value=True)
    # debounce_sec = st.sidebar.slider("Debounce (seconds)", 0.0, 3.0, 1.0, 0.1)

    # now = time.time()

    # # mỗi lần chỉnh editor, Streamlit rerun -> ta lưu timestamp
    # st.session_state["last_edit_ts"] = now

    # # Sau đó ở dưới (main page), chỉ solve khi đã "im" đủ debounce_sec
    # if auto_run:
    #     last_ts = st.session_state.get("last_edit_ts", 0.0)
    #     if time.time() - last_ts >= debounce_sec:
    #         # solve
    #         pass
    #     else:
    #         st.caption("Waiting for inputs to settle...")


    # -------------------------
    # 3) Build & solve model
    # -------------------------
    def build_avail_dict(avail_df: pd.DataFrame):
        """
        avail_df dạng wide:
        - columns: Mon, Tue, Wed, Thu, Fri
        - index: operator codes (EK, YC, AT,...)
        """
        days_cols = [c.strip() for c in avail_df.columns]  # ['Mon','Tue',...]
        tmp = avail_df.copy()
        tmp.columns = days_cols

        avail = {}
        # operator nằm ở index
        for op in tmp.index.astype(str):
            for d in days_cols:
                val = tmp.loc[op, d]
                hrs = float(val) if pd.notna(val) else 0.0
                avail[(op, d)] = hrs
        return avail


    def solve_model(opt_mode: str, avail_df: pd.DataFrame, use_skill: bool, skill_req: float, skill_map: dict):

        avail = build_avail_dict(avail_df)

        x = pulp.LpVariable.dicts("hours", (ops, days), lowBound=0, cat="Continuous")

        if opt_mode == "Minimize cost":
            model = pulp.LpProblem("Workforce_Cost_Min", pulp.LpMinimize)
            model += pulp.lpSum(wage_ui[o] * x[o][d] for o in ops for d in days)
        else:
            model = pulp.LpProblem("Workforce_Fairness_MinRange", pulp.LpMinimize)
            weekly_hours = {o: pulp.lpSum(x[o][d] for d in days) for o in ops}
            max_hours = pulp.LpVariable("max_hours", lowBound=0)
            min_hours_var = pulp.LpVariable("min_hours", lowBound=0)
            for o in ops:
                model += weekly_hours[o] <= max_hours
                model += weekly_hours[o] >= min_hours_var
            model += (max_hours - min_hours_var)

        # (1) Daily coverage
        for d in days:
            model += pulp.lpSum(x[o][d] for o in ops) == int(coverage_per_day), f"coverage_{d}"

        # (2) Min weekly hours
        for o in ops:
            model += pulp.lpSum(x[o][d] for d in days) >= int(min_week[o]), f"minweek_{o}"

        # (3) Availability
        for o in ops:
            for d in days:
                cap = avail.get((o, d), 0)
                model += x[o][d] <= cap, f"avail_{o}_{d}"

        # (4) Skill constraints (optional)
        
        if use_skill and skill_req > 0:
            for d in days:
                for sk in skills:
                    model += pulp.lpSum(
                        x[o][d] * skill_map.get((o, sk), 0)
                        for o in ops
                    ) >= float(skill_req), f"skill_{sk}_{d}"


        status = model.solve(pulp.PULP_CBC_CMD(msg=0))
        return model, x, status
    

    auto_run = st.sidebar.checkbox("⚡ Auto-run optimization", value=True)

    if auto_run and need_solve:
        st.session_state["last_settings_key"] = settings_key

        # --- CALL SOLVER (đúng signature) ---
        model, x, status = solve_model(
            opt_mode=opt_mode,
            avail_df=st.session_state["avail_df"],
            use_skill=use_skill,
            skill_req=skill_req,
            skill_map=skill_map_ui
        )


        # --- SAVE RESULT ---
        st.session_state["opt_result"] = {
            "model": model,
            "x": x,
            "status": status,
            "objective": pulp.value(model.objective) if model is not None else None
        }


    st.subheader("✅ Solver result")

    if "opt_result" not in st.session_state:
        st.info("No result yet. Change constraints to run automatically.")
    else:
        res = st.session_state["opt_result"]
        st.write("Status:", pulp.LpStatus[res["status"]])
        if res["objective"] is not None:
            st.write("Objective value:", round(res["objective"], 4))



        # -------------------------
        # 4) Build schedule table
        # -------------------------
        st.subheader("📌 Optimized hours table")

        if "opt_result" not in st.session_state or st.session_state["opt_result"] is None:
            st.info("No solution yet (change constraints to auto-run, or run solver).")
        else:
            res = st.session_state["opt_result"]
            x_var = res["x"]          # ✅ lấy x từ session_state
            status = res["status"]

            if pulp.LpStatus[status] != "Optimal":
                st.warning(f"Solver status: {pulp.LpStatus[status]} (table may be incomplete)")
            # hours_df: rows = operators, cols = days
            hours_df = pd.DataFrame(
                {d: [float(x_var[o][d].value() or 0) for o in ops]
                for d in days},
                index=ops
            )
            hours_df.index.name = "Operator"
            st.dataframe(hours_df, use_container_width=True)


        import numpy as np  # đảm bảo có ở đầu file (nếu bạn chưa import)

        # -------------------------
        # 5) Chart: total hours per operator (week)
        # -------------------------


        # -------------------------
        # Prepare data
        # -------------------------

        # sched: scheduled hours từ solver
        sched = hours_df.copy()
        sched["Total_scheduled"] = sched[days].sum(axis=1)

        # avail: availability từ sidebar
        avail_wide = st.session_state["avail_df"].copy()
        avail_wide = avail_wide.apply(pd.to_numeric, errors="coerce").fillna(0)
        avail_wide["Total_available"] = avail_wide[days].sum(axis=1)

        # merge để vẽ & làm bảng
        summary = pd.DataFrame(index=ops)
        summary["Total_available"] = avail_wide.loc[ops, "Total_available"]
        summary["Total_scheduled"] = sched.loc[ops, "Total_scheduled"]

        summary["Utilization_%"] = (
            summary["Total_scheduled"]
            / summary["Total_available"].replace(0, np.nan)
            * 100
        ).round(2)

        # -------------------------
        # Layout: chart LEFT, table RIGHT
        # -------------------------
        st.divider()
        col_chart, col_table = st.columns([4, 3])

        # -------------------------
        # Chart (LEFT)
        # -------------------------
        with col_chart:
            st.markdown("### 📊 Weekly working hours: Scheduled vs Available")
            fig, ax = plt.subplots(figsize=(10, 4))

            x = np.arange(len(summary.index))
            width = 0.35

            ax.bar(
                x - width / 2,
                summary["Total_scheduled"],
                width,
                label="Scheduled hours",
                color="#4C72B0"
            )

            ax.bar(
                x + width / 2,
                summary["Total_available"],
                width,
                label="Available hours",
                color="#DD8452"
            )

            ax.set_xticks(x)
            ax.set_xticklabels(summary.index.astype(str))
            ax.set_xlabel("Operator / Student")
            ax.set_ylabel("Hours per week")
            ax.set_title("Scheduled vs Available hours per operator (week)")

            ax.legend(frameon=False)
            st.pyplot(fig)

        # -------------------------
        # Table (RIGHT)
        # -------------------------
        with col_table:
            st.markdown("### Summary table")
            st.dataframe(
                summary.reset_index()
                .rename(columns={"index": "Operator"}),
                use_container_width=True
            )


        # -------------------------
        # 6) Optional: stacked by day (distribution across days)
        # -------------------------
        st.divider()
        st.subheader("📊 Stacked chart")

        view_mode = st.radio(
            "View stacked chart by:",
            ["By day", "By operator (week)"],
            horizontal=False,
            key="stack_view_mode"
        )

        if view_mode == "By day":
            # ===== Horizontal stacked: each bar = a DAY, stacked by operators =====
            fig2, ax2 = plt.subplots(figsize=(10, 4))

            left = np.zeros(len(days))  # stack baseline per day

            for o in ops:
                vals = sched.loc[o, days].values  # hours of operator o across days
                ax2.barh(days, vals, left=left, label=o)
                left += vals

            ax2.set_xlabel("Hours")   # ✅ hours is horizontal axis
            ax2.set_ylabel("Day")     # ✅ days are on y-axis
            ax2.set_title("Daily working hours breakdown by operator")

            ax2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=3,
                frameon=False
            )
            st.pyplot(fig2)

        else:
            # ===== Vertical stacked: each bar = an OPERATOR, stacked by days =====
            fig2, ax2 = plt.subplots(figsize=(10, 4))

            bottom = np.zeros(len(ops))  # stack baseline per operator

            for d in days:
                vals = sched[d].values  # hours of all operators on day d
                ax2.barh(ops, vals, left=bottom, label=d)
                bottom += vals

            ax2.set_xlabel("Operator / Student")
            ax2.set_ylabel("Hours")
            ax2.set_title("Weekly allocation breakdown by day")

            ax2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=len(days),
                frameon=False
            )
            st.pyplot(fig2)

        #numerical vs numericl: median line plot
        #numerical vs categorical: box plot
        #categorical vs categorical: heatmap



        
            



