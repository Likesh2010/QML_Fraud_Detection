import os

import numpy as np

import pandas as pd

import streamlit as st



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.svm import OneClassSVM

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    average_precision_score,

)



from qiskit.circuit.library import zz_feature_map

from qiskit_machine_learning.kernels import FidelityQuantumKernel





# ============================================================

# CONFIGURATION

# ============================================================



DATA_PATH = "data/creditcard.csv"
SAMPLE_DATA_PATH = "data/creditcard_sample.csv"

RESULTS_DIR = "results"



# ============================================================

# MODEL CONFIGURATIONS

# ============================================================



# 500-sample reference experiment

QML_500_N_TRAIN = 500

QML_500_REPS = 1

QML_500_NU = 0.15

_qml_500_alt_path = os.path.join(
    RESULTS_DIR,
    "quantum kernels",
    "reps1_linear_K_train.npy"
)

QML_500_KERNEL_PATH = (
    _qml_500_alt_path
    if os.path.exists(_qml_500_alt_path)
    else os.path.join(RESULTS_DIR, "reps1_linear_K_train.npy")
)



# 700-sample experiment / active prototype model

QML_700_N_TRAIN = 700

QML_700_REPS = 2

QML_700_NU = 0.08

QML_700_KERNEL_PATH = os.path.join(

    RESULTS_DIR,

    "training_size_experiment",

    "700_reps2",

    "K_train.npy"

)



# Common quantum configuration

N_QUBITS = 4

ENTANGLEMENT = "linear"



# Active prototype configuration = 700-sample model

REPS = QML_700_REPS

NU = QML_700_NU



# Data configuration

TRAIN_RATIO = 0.70

N_TRAIN = QML_700_N_TRAIN

N_TEST_NORMAL = 500

RANDOM_STATE = 42



# Original model features

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]





# ============================================================

# PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="Quantum Fraud Detection",

    page_icon="⚛️",

    layout="wide",

)





# ============================================================

# CUSTOM CSS

# ============================================================



st.markdown(
    """
    <style>
    .main-title { font-size: 2.55rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.15rem; }
    .subtitle { font-size: 1.05rem; color: #667085; margin-bottom: 1.25rem; }
    .hero { padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.18); border-radius: 16px; background: linear-gradient(135deg, rgba(99,102,241,.10), rgba(14,165,233,.07)); margin-bottom: 1.25rem; }
    .badge { display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; border: 1px solid rgba(99,102,241,.25); background: rgba(99,102,241,.10); }
    .section-note { padding: 0.75rem 1rem; border-left: 4px solid #6366f1; background: rgba(99,102,241,.06); border-radius: 0 10px 10px 0; margin: 0.5rem 0 1rem 0; }
    div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.18); border-radius: 14px; padding: 0.8rem 1rem; background: rgba(128,128,128,.035); }
    div[data-testid="stMetricLabel"] { font-size: 0.82rem; }
    .stButton > button { border-radius: 10px; font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)


# HEADER

# ============================================================



st.markdown(
    '<div class="hero">'
    '<div class="main-title">⚛️ Quantum Fraud Detection</div>'
    '<div class="subtitle">Quantum-Kernel One-Class SVM for Credit Card Fraud Detection</div>'
    '<span class="badge">ACTIVE PROTOTYPE • 700 NORMAL TRAINING SAMPLES</span>'
    '</div>',
    unsafe_allow_html=True,
)


# LOAD DATASET

# ============================================================



@st.cache_data
def load_dataset():
    target_path = DATA_PATH if os.path.exists(DATA_PATH) else SAMPLE_DATA_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}' or '{SAMPLE_DATA_PATH}'."
        )

    df = pd.read_csv(target_path)



    # Sort chronologically

    df = df.sort_values("Time").reset_index(drop=True)



    # Same transformation used during training

    df["Amount"] = np.log1p(df["Amount"])



    return df





# ============================================================

# PREPROCESSING

# ============================================================



@st.cache_resource

def prepare_preprocessing():



    df = load_dataset()



    # --------------------------------------------------------

    # TIME-BASED SPLIT

    # --------------------------------------------------------



    split_index = int(

        len(df) * TRAIN_RATIO

    )



    train_df = df.iloc[

        :split_index

    ].copy()



    test_df = df.iloc[

        split_index:

    ].copy()



    # --------------------------------------------------------

    # NORMAL-ONLY TRAINING

    # --------------------------------------------------------



    normal_train_df = train_df[

        train_df["Class"] == 0

    ].copy()



    train_df = normal_train_df.sample(

        n=N_TRAIN,

        random_state=RANDOM_STATE

    )



    # --------------------------------------------------------

    # TEST DATA

    # --------------------------------------------------------



    normal_test_df = test_df[

        test_df["Class"] == 0

    ].copy()



    fraud_test_df = test_df[

        test_df["Class"] == 1

    ].copy()



    normal_test_df = normal_test_df.sample(

        n=min(

            N_TEST_NORMAL,

            len(normal_test_df)

        ),

        random_state=RANDOM_STATE

    )



    diagnostic_test_df = pd.concat(

        [

            normal_test_df,

            fraud_test_df

        ],

        axis=0

    )



    diagnostic_test_df = diagnostic_test_df.sample(

        frac=1,

        random_state=RANDOM_STATE

    ).reset_index(drop=True)



    # --------------------------------------------------------

    # RAW FEATURES

    # --------------------------------------------------------



    X_train_raw = train_df[

        FEATURES

    ].values



    X_test_raw = diagnostic_test_df[

        FEATURES

    ].values



    # --------------------------------------------------------

    # STANDARD SCALING

    # --------------------------------------------------------



    scaler = StandardScaler()



    X_train_scaled = scaler.fit_transform(

        X_train_raw

    )



    X_test_scaled = scaler.transform(

        X_test_raw

    )



    # --------------------------------------------------------

    # PCA: 29 → 4

    # --------------------------------------------------------



    pca = PCA(

        n_components=N_QUBITS

    )



    X_train_pca = pca.fit_transform(

        X_train_scaled

    )



    X_test_pca = pca.transform(

        X_test_scaled

    )



    # --------------------------------------------------------

    # QUANTUM ANGLE SCALING

    # --------------------------------------------------------



    angle_scaler = MinMaxScaler(

        feature_range=(0, np.pi)

    )



    X_train = angle_scaler.fit_transform(

        X_train_pca

    )



    X_test = angle_scaler.transform(

        X_test_pca

    )



    X_test = np.clip(

        X_test,

        0,

        np.pi

    )



    return (

        df,

        train_df,

        diagnostic_test_df,

        scaler,

        pca,

        angle_scaler,

        X_train,

        X_test,

    )





# ============================================================

# QUANTUM KERNEL

# ============================================================



@st.cache_resource

def create_quantum_kernel():



    feature_map = zz_feature_map(

        feature_dimension=N_QUBITS,

        reps=REPS,

        entanglement=ENTANGLEMENT,

    )



    kernel = FidelityQuantumKernel(

        feature_map=feature_map

    )



    return kernel





# ============================================================

# LOAD CACHED TRAINING KERNELS

# ============================================================



@st.cache_data

def load_training_kernel(model_version="700"):



    if model_version == "500":

        path = QML_500_KERNEL_PATH

    else:

        path = QML_700_KERNEL_PATH



    if not os.path.exists(path):

        raise FileNotFoundError(

            "Cached QML training kernel not found:\n"

            f"{path}"

        )



    return np.load(path)





# ============================================================

# TRAIN ONE-CLASS SVM

# ============================================================



@st.cache_resource

def train_qml_model(model_version="700"):



    if model_version == "500":

        nu = QML_500_NU

    else:

        nu = QML_700_NU



    K_train = load_training_kernel(model_version)



    model = OneClassSVM(

        kernel="precomputed",

        nu=nu

    )



    model.fit(K_train)



    return model





# ============================================================

# TRAIN CLASSICAL ONE-CLASS SVM

# ============================================================



@st.cache_resource

def train_classical_model(model_version="700"):



    nu = (

        QML_500_NU

        if model_version == "500"

        else QML_700_NU

    )



    (

        _,

        train_df,

        _,

        scaler,

        pca,

        _,

        _,

        _,

    ) = prepare_model_preprocessing(

        model_version

    )



    X_train_raw = train_df[

        FEATURES

    ].values



    X_train_scaled = scaler.transform(

        X_train_raw

    )



    X_train_pca = pca.transform(

        X_train_scaled

    )



    model = OneClassSVM(

        kernel="rbf",

        nu=nu,

        gamma="scale"

    )



    model.fit(

        X_train_pca

    )



    return model





# ============================================================

# MODEL CONFIG HELPER

# ============================================================



def get_model_config(model_version="700"):



    if model_version == "500":

        return {

            "label": "QML — 500 Samples",

            "n_train": QML_500_N_TRAIN,

            "reps": QML_500_REPS,

            "nu": QML_500_NU,

        }



    return {

        "label": "QML — 700 Samples",

        "n_train": QML_700_N_TRAIN,

        "reps": QML_700_REPS,

        "nu": QML_700_NU,

    }





# ============================================================

# MODEL-SPECIFIC PREPROCESSING

# ============================================================



@st.cache_resource

def prepare_model_preprocessing(model_version="700"):



    df = load_dataset()



    if "Split" in df.columns:
        train_df = df[df["Split"] == "train"].copy()
        test_df = df[df["Split"] == "test"].copy()
        normal_train_df = train_df[train_df["Class"] == 0].copy()
        n_train = (
            QML_500_N_TRAIN
            if model_version == "500"
            else min(QML_700_N_TRAIN, len(normal_train_df))
        )
        selected_train_df = normal_train_df.sample(
            n=n_train,
            random_state=RANDOM_STATE
        )
        diagnostic_test_df = test_df.sample(
            frac=1,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)
    else:
        split_index = int(
            len(df) * TRAIN_RATIO
        )

        train_df = df.iloc[
            :split_index
        ].copy()

        test_df = df.iloc[
            split_index:
        ].copy()

        normal_train_df = train_df[
            train_df["Class"] == 0
        ].copy()

        n_train = (
            QML_500_N_TRAIN
            if model_version == "500"
            else QML_700_N_TRAIN
        )

        selected_train_df = normal_train_df.sample(
            n=n_train,
            random_state=RANDOM_STATE
        )

        normal_test_df = test_df[
            test_df["Class"] == 0
        ].copy()

        fraud_test_df = test_df[
            test_df["Class"] == 1
        ].copy()

        normal_test_df = normal_test_df.sample(
            n=min(
                N_TEST_NORMAL,
                len(normal_test_df)
            ),
            random_state=RANDOM_STATE
        )

        diagnostic_test_df = pd.concat(
            [
                normal_test_df,
                fraud_test_df
            ],
            axis=0
        )

        diagnostic_test_df = diagnostic_test_df.sample(
            frac=1,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)



    X_train_raw = selected_train_df[

        FEATURES

    ].values



    X_test_raw = diagnostic_test_df[

        FEATURES

    ].values



    scaler = StandardScaler()



    X_train_scaled = scaler.fit_transform(

        X_train_raw

    )



    X_test_scaled = scaler.transform(

        X_test_raw

    )



    pca = PCA(

        n_components=N_QUBITS

    )



    X_train_pca = pca.fit_transform(

        X_train_scaled

    )



    X_test_pca = pca.transform(

        X_test_scaled

    )



    angle_scaler = MinMaxScaler(

        feature_range=(0, np.pi)

    )



    X_train = angle_scaler.fit_transform(

        X_train_pca

    )



    X_test = angle_scaler.transform(

        X_test_pca

    )



    X_test = np.clip(

        X_test,

        0,

        np.pi

    )



    return (

        df,

        selected_train_df,

        diagnostic_test_df,

        scaler,

        pca,

        angle_scaler,

        X_train,

        X_test,

    )





# Compatibility wrapper: active prototype = 700 samples

@st.cache_resource

def prepare_preprocessing():



    return prepare_model_preprocessing("700")





# ============================================================

# QUANTUM KERNEL

# ============================================================



@st.cache_resource

def create_quantum_kernel(reps=2):



    feature_map = zz_feature_map(

        feature_dimension=N_QUBITS,

        reps=reps,

        entanglement=ENTANGLEMENT

    )



    kernel = FidelityQuantumKernel(

        feature_map=feature_map

    )



    return kernel





# ============================================================

# SIDEBAR

# ============================================================



st.sidebar.title("Navigation")



page = st.sidebar.radio(

    "Select page",

    [

        "Dashboard",

        "Demo Prediction",

        "Live Prediction",

        "Model Results",

        "Graphs",

    ],

)





# ============================================================

st.sidebar.divider()
st.sidebar.caption("ACTIVE MODEL")
st.sidebar.success("700-sample QML")
st.sidebar.caption("4 qubits • ZZ • reps=2 • ν=0.08")
st.sidebar.caption("1000-sample results are shown as a training-size experiment; they do not change the active prediction model.")


# DASHBOARD

# ============================================================



if page == "Dashboard":



    df = load_dataset()

    using_sample = not os.path.exists(DATA_PATH)
    if using_sample:
        total_transactions = 284807
        fraud_count = 492
        normal_count = 284315
        fraud_rate = (fraud_count / total_transactions) * 100
    else:
        total_transactions = len(df)
        fraud_count = int((df["Class"] == 1).sum())
        normal_count = int((df["Class"] == 0).sum())
        fraud_rate = (fraud_count / total_transactions) * 100



    st.header("Project Dashboard")



    col1, col2, col3, col4 = st.columns(4)



    col1.metric(

        "Transactions",

        f"{total_transactions:,}"

    )



    col2.metric(

        "Normal",

        f"{normal_count:,}"

    )



    col3.metric(

        "Fraud",

        f"{fraud_count:,}"

    )



    col4.metric(

        "Fraud Rate",

        f"{fraud_rate:.4f}%"

    )

    st.markdown(
        '<div class="section-note"><b>Active model:</b> 700-sample QML • 4 qubits • ZZ feature map • reps=2 • linear entanglement • Fidelity Quantum Kernel • One-Class SVM ν=0.08</div>',
        unsafe_allow_html=True,
    )



    st.divider()



    # --------------------------------------------------------

    # MODEL CONFIGURATION

    # --------------------------------------------------------



    st.subheader(

        "Quantum Model Configuration"

    )



    config = pd.DataFrame(

        {

            "Parameter": [

                "Algorithm",

                "Original Features",

                "Qubits",

                "Feature Map",

                "Repetitions",

                "Entanglement",

                "Quantum Kernel",

                "Anomaly Detector",

                "ν",

            ],

            "QML — 500 Samples": [

                "Quantum Kernel One-Class SVM",

                "V1–V28 + Amount = 29",

                "4",

                "ZZ",

                "1",

                "Linear",

                "Fidelity Quantum Kernel",

                "One-Class SVM",

                "0.15",

            ],

            "QML — 700 Samples": [

                "Quantum Kernel One-Class SVM",

                "V1–V28 + Amount = 29",

                "4",

                "ZZ",

                "2",

                "Linear",

                "Fidelity Quantum Kernel",

                "One-Class SVM",

                "0.08",

            ],

        }

    )



    st.table(config)



    st.info(

        "The 500-sample model is the reference experiment. "

        "The 700-sample model is the active prototype configuration."

    )



    st.divider()



    # --------------------------------------------------------

    # PIPELINE

    # --------------------------------------------------------



    st.subheader("Pipeline")



    st.code(

        """

Credit Card Transaction

          ↓

Time-Based Split

          ↓

Normal-Only Training

          ↓

StandardScaler

          ↓

PCA: 29 → 4

          ↓

Scale to [0, π]

          ↓

ZZ Quantum Feature Map

          ↓

Fidelity Quantum Kernel

          ↓

One-Class SVM

          ↓

Normal / Anomaly

        """,

        language="text",

    )



    st.divider()



    st.subheader(

        "Why Quantum Kernel?"

    )



    st.write(

        "The system uses a quantum feature map to encode "

        "the four PCA components into a quantum state. "

        "The Fidelity Quantum Kernel measures similarity "

        "between quantum-encoded transactions."

    )





# ============================================================

# DEMO PREDICTION

# ============================================================



elif page == "Demo Prediction":



    st.header("Demo Transaction Prediction")



    st.info(

        "The selected transaction is evaluated separately using the "

        "500-sample and 700-sample experiments. Each size contains "

        "one QML model and one classical RBF One-Class SVM baseline."

    )



    # --------------------------------------------------------

    # PREPARE BOTH TRAINING-SIZE EXPERIMENTS

    # --------------------------------------------------------



    (

        _,

        train_df_500,

        test_df,

        scaler_500,

        pca_500,

        _,

        X_train_500,

        X_test_500,

    ) = prepare_model_preprocessing("500")



    (

        _,

        train_df_700,

        test_df_700,

        scaler_700,

        pca_700,

        _,

        X_train_700,

        X_test_700,

    ) = prepare_model_preprocessing("700")



    # Both diagnostic test sets are constructed identically.

    # Keep the 500-sample test frame as the reference for the

    # transaction selector.

    if len(test_df) != len(test_df_700):

        st.error(

            "The 500-sample and 700-sample test sets do not match."

        )

        st.stop()



    model_500 = train_qml_model("500")

    model_700 = train_qml_model("700")



    classical_500 = train_classical_model("500")

    classical_700 = train_classical_model("700")



    kernel_500 = create_quantum_kernel(QML_500_REPS)

    kernel_700 = create_quantum_kernel(QML_700_REPS)



    transaction_number = st.number_input(

        "Select transaction",

        min_value=0,

        max_value=len(test_df) - 1,

        value=0,

        step=1,

    )



    selected_row = test_df.iloc[

        transaction_number

    ]



    actual_class = int(

        selected_row["Class"]

    )



    col1, col2, col3 = st.columns(3)



    col1.metric(

        "Transaction Index",

        transaction_number

    )



    col2.metric(

        "Amount",

        f"${np.expm1(selected_row['Amount']):.2f}"

    )



    col3.metric(

        "Actual Class",

        "Fraud" if actual_class == 1 else "Normal"

    )



    st.divider()



    if st.button(

        "Run All Four Models",

        type="primary"

    ):



        # ----------------------------------------------------

        # QUANTUM REPRESENTATIONS

        # ----------------------------------------------------



        x_500 = X_test_500[

            transaction_number

        ].reshape(1, -1)



        x_700 = X_test_700[

            transaction_number

        ].reshape(1, -1)



        # ----------------------------------------------------

        # CLASSICAL REPRESENTATIONS

        # Same scaler + PCA pipeline, but without quantum

        # angle scaling.

        # ----------------------------------------------------



        raw_transaction = selected_row[

            FEATURES

        ].values.reshape(1, -1)



        x_500_classical = pca_500.transform(

            scaler_500.transform(

                raw_transaction

            )

        )



        x_700_classical = pca_700.transform(

            scaler_700.transform(

                raw_transaction

            )

        )



        # ----------------------------------------------------

        # QUANTUM KERNEL EVALUATION

        # ----------------------------------------------------



        with st.spinner(

            "Calculating QML predictions..."

        ):



            K_500 = kernel_500.evaluate(

                x_vec=x_500,

                y_vec=X_train_500

            )



            K_700 = kernel_700.evaluate(

                x_vec=x_700,

                y_vec=X_train_700

            )



        # ----------------------------------------------------

        # QML 500

        # ----------------------------------------------------



        raw_qml_500 = model_500.predict(

            K_500

        )[0]



        score_qml_500 = model_500.decision_function(

            K_500

        )[0]



        pred_qml_500 = (

            1 if raw_qml_500 == -1 else 0

        )



        # ----------------------------------------------------

        # QML 700

        # ----------------------------------------------------



        raw_qml_700 = model_700.predict(

            K_700

        )[0]



        score_qml_700 = model_700.decision_function(

            K_700

        )[0]



        pred_qml_700 = (

            1 if raw_qml_700 == -1 else 0

        )



        # ----------------------------------------------------

        # CLASSICAL 500

        # ----------------------------------------------------



        raw_classical_500 = classical_500.predict(

            x_500_classical

        )[0]



        score_classical_500 = (

            classical_500.decision_function(

                x_500_classical

            )[0]

        )



        pred_classical_500 = (

            1 if raw_classical_500 == -1 else 0

        )



        # ----------------------------------------------------

        # CLASSICAL 700

        # ----------------------------------------------------



        raw_classical_700 = classical_700.predict(

            x_700_classical

        )[0]



        score_classical_700 = (

            classical_700.decision_function(

                x_700_classical

            )[0]

        )



        pred_classical_700 = (

            1 if raw_classical_700 == -1 else 0

        )



        # ----------------------------------------------------

        # 500-SAMPLE RESULTS

        # ----------------------------------------------------



        st.subheader("500-Sample Experiment")



        st.caption(

            "QML: 500 normal training samples, ZZ, 4 qubits, "

            "reps=1, ν=0.15. Classical: 500 normal training "

            "samples, 4D PCA, RBF One-Class SVM, ν=0.15."

        )



        result_500 = pd.DataFrame(

            {

                "Model": [

                    "QML — 500 Samples",

                    "Classical — 500 Samples",

                ],

                "Method": [

                    "Fidelity Quantum Kernel",

                    "RBF Kernel",

                ],

                "Prediction": [

                    (

                        "FRAUD / ANOMALY"

                        if pred_qml_500 == 1

                        else "NORMAL"

                    ),

                    (

                        "FRAUD / ANOMALY"

                        if pred_classical_500 == 1

                        else "NORMAL"

                    ),

                ],

                "Anomaly Score": [

                    score_qml_500,

                    score_classical_500,

                ],

                "Raw OCSVM": [

                    raw_qml_500,

                    raw_classical_500,

                ],

                "Correct": [

                    pred_qml_500 == actual_class,

                    pred_classical_500 == actual_class,

                ],

            }

        )



        st.dataframe(

            result_500,

            use_container_width=True,

            hide_index=True

        )



        # ----------------------------------------------------

        # 700-SAMPLE RESULTS

        # ----------------------------------------------------



        st.divider()



        st.subheader("700-Sample Experiment")



        st.caption(

            "QML: 700 normal training samples, ZZ, 4 qubits, "

            "reps=2, ν=0.08. Classical: 700 normal training "

            "samples, 4D PCA, RBF One-Class SVM, ν=0.08."

        )



        result_700 = pd.DataFrame(

            {

                "Model": [

                    "QML — 700 Samples",

                    "Classical — 700 Samples",

                ],

                "Method": [

                    "Fidelity Quantum Kernel",

                    "RBF Kernel",

                ],

                "Prediction": [

                    (

                        "FRAUD / ANOMALY"

                        if pred_qml_700 == 1

                        else "NORMAL"

                    ),

                    (

                        "FRAUD / ANOMALY"

                        if pred_classical_700 == 1

                        else "NORMAL"

                    ),

                ],

                "Anomaly Score": [

                    score_qml_700,

                    score_classical_700,

                ],

                "Raw OCSVM": [

                    raw_qml_700,

                    raw_classical_700,

                ],

                "Correct": [

                    pred_qml_700 == actual_class,

                    pred_classical_700 == actual_class,

                ],

            }

        )



        st.dataframe(

            result_700,

            use_container_width=True,

            hide_index=True

        )



        st.divider()



        st.success(

            "All four models were evaluated on the same selected transaction. "

            "The 700-sample QML model remains the active prototype model."

        )



# ============================================================

# LIVE PREDICTION

# ============================================================



elif page == "Live Prediction":



    st.header(

        "Live Transaction Prediction"

    )



    st.info(

        "Upload a CSV containing V1–V28 and Amount. "

        "If the CSV also contains Class, the actual class "

        "will also be displayed for verification."

    )



    st.warning(

        "Quantum-kernel evaluation is computationally expensive. "

        "The prototype processes a maximum of 5 uploaded "

        "transactions at a time."

    )



    st.info(

        "Active prototype model: QML — 700 Samples | "

        "4 qubits | ZZ | reps=2 | linear | ν=0.08. "

        "The 500-sample reference model remains available "

        "for comparison in Demo Prediction and Model Results."

    )



    (

        df,

        train_df,

        test_df,

        scaler,

        pca,

        angle_scaler,

        X_train,

        X_test,

    ) = prepare_preprocessing()



    model = train_qml_model()



    kernel = create_quantum_kernel()



    # --------------------------------------------------------

    # FILE UPLOAD

    # --------------------------------------------------------



    uploaded_file = st.file_uploader(

        "Upload transaction CSV",

        type=["csv"]

    )



    if uploaded_file is not None:



        try:



            input_df = pd.read_csv(

                uploaded_file

            )



            st.subheader(

                "Uploaded Transaction Data"

            )



            st.dataframe(

                input_df.head(5),

                use_container_width=True

            )



            # ------------------------------------------------

            # CHECK REQUIRED FEATURES

            # ------------------------------------------------



            missing_features = [

                feature

                for feature in FEATURES

                if feature not in input_df.columns

            ]



            if missing_features:



                st.error(

                    "Required columns are missing:"

                )



                st.write(

                    missing_features

                )



            else:



                # --------------------------------------------

                # LIMIT INPUT

                # --------------------------------------------



                if len(input_df) > 5:



                    st.warning(

                        "Only the first 5 transactions "

                        "will be processed."

                    )



                    working_df = (

                        input_df.head(5).copy()

                    )



                else:



                    working_df = input_df.copy()



                st.write(

                    f"Transactions to process: "

                    f"**{len(working_df)}**"

                )



                # --------------------------------------------

                # RUN BUTTON

                # --------------------------------------------



                if st.button(

                    "Run Quantum Detection",

                    type="primary"

                ):



                    try:



                        # ====================================

                        # PREPROCESSING

                        # ====================================



                        raw_data = working_df[

                            FEATURES

                        ].copy()



                        # Same Amount transformation

                        # used during model training

                        raw_data["Amount"] = (

                            np.log1p(

                                raw_data["Amount"]

                            )

                        )



                        # StandardScaler

                        scaled_data = (

                            scaler.transform(

                                raw_data.values

                            )

                        )



                        # PCA 29 → 4

                        pca_data = (

                            pca.transform(

                                scaled_data

                            )

                        )



                        # Scale to quantum angles

                        quantum_data = (

                            angle_scaler.transform(

                                pca_data

                            )

                        )



                        quantum_data = np.clip(

                            quantum_data,

                            0,

                            np.pi

                        )



                        # ====================================

                        # QUANTUM KERNEL

                        # ====================================



                        with st.spinner(

                            "Calculating quantum kernel..."

                        ):



                            K_new = (

                                kernel.evaluate(

                                    x_vec=quantum_data,

                                    y_vec=X_train

                                )

                            )



                            # Raw OCSVM output:

                            # +1 = normal

                            # -1 = anomaly

                            raw_predictions = (

                                model.predict(K_new)

                            )



                            # Higher score = more normal

                            decision_scores = (

                                model.decision_function(

                                    K_new

                                )

                            )



                        # ====================================

                        # CONVERT OUTPUT

                        # ====================================



                        predicted_classes = np.where(

                            raw_predictions == -1,

                            1,

                            0

                        )



                        # ====================================

                        # RESULT DATAFRAME

                        # ====================================



                        result_df = (

                            working_df.copy()

                        )



                        result_df[

                            "Prediction"

                        ] = np.where(

                            predicted_classes == 1,

                            "FRAUD / ANOMALY",

                            "NORMAL"

                        )



                        result_df[

                            "Anomaly Score"

                        ] = decision_scores



                        result_df[

                            "Raw OCSVM Output"

                        ] = raw_predictions



                        # ====================================

                        # ACTUAL CLASS IF AVAILABLE

                        # ====================================



                        actual_classes = None



                        if "Class" in working_df.columns:



                            actual_classes = (

                                working_df[

                                    "Class"

                                ]

                                .astype(int)

                                .values

                            )



                            result_df[

                                "Actual Class"

                            ] = np.where(

                                actual_classes == 1,

                                "FRAUD",

                                "NORMAL"

                            )



                            result_df[

                                "Correct"

                            ] = (

                                predicted_classes

                                == actual_classes

                            )



                        # ====================================

                        # SUCCESS

                        # ====================================



                        st.success(

                            "Quantum detection completed."

                        )



                        # ====================================

                        # SUMMARY METRICS

                        # ====================================



                        fraud_detected = int(

                            np.sum(

                                predicted_classes == 1

                            )

                        )



                        normal_detected = int(

                            np.sum(

                                predicted_classes == 0

                            )

                        )



                        col1, col2, col3 = (

                            st.columns(3)

                        )



                        col1.metric(

                            "Transactions Processed",

                            len(working_df)

                        )



                        col2.metric(

                            "Normal",

                            normal_detected

                        )



                        col3.metric(

                            "Anomalies",

                            fraud_detected

                        )



                        # ====================================

                        # PREDICTION TABLE

                        # ====================================



                        st.divider()



                        st.subheader(

                            "Quantum Model Predictions"

                        )



                        st.dataframe(

                            result_df,

                            use_container_width=True

                        )



                        # ====================================

                        # VERIFICATION

                        # ====================================



                        if actual_classes is not None:



                            st.divider()



                            st.subheader(

                                "Prediction Verification"

                            )



                            verification_accuracy = (

                                accuracy_score(

                                    actual_classes,

                                    predicted_classes

                                )

                            )



                            st.metric(

                                "Accuracy on Uploaded Rows",

                                f"{verification_accuracy * 100:.2f}%"

                            )



                            cm = pd.DataFrame(

                                {

                                    "Metric": [

                                        "Correct",

                                        "Incorrect",

                                        "Actual Fraud",

                                        "Predicted Fraud",

                                    ],

                                    "Value": [

                                        int(

                                            np.sum(

                                                predicted_classes

                                                == actual_classes

                                            )

                                        ),

                                        int(

                                            np.sum(

                                                predicted_classes

                                                != actual_classes

                                            )

                                        ),

                                        int(

                                            np.sum(

                                                actual_classes == 1

                                            )

                                        ),

                                        int(

                                            np.sum(

                                                predicted_classes == 1

                                            )

                                        ),

                                    ],

                                }

                            )



                            st.table(cm)



                            st.caption(

                                "This verification is only for "

                                "the uploaded transactions."

                            )



                        # ====================================

                        # TECHNICAL DETAILS

                        # ====================================



                        st.divider()



                        with st.expander(

                            "Technical Model Output"

                        ):



                            st.write(

                                "Original features:",

                                29

                            )



                            st.write(

                                "PCA components:",

                                N_QUBITS

                            )



                            st.write(

                                "Quantum feature map:",

                                "ZZ"

                            )



                            st.write(

                                "Qubits:",

                                N_QUBITS

                            )



                            st.write(

                                "Repetitions:",

                                REPS

                            )



                            st.write(

                                "Entanglement:",

                                ENTANGLEMENT

                            )



                            st.write(

                                "Quantum kernel:",

                                "Fidelity Quantum Kernel"

                            )



                            st.write(

                                "One-Class SVM ν:",

                                NU

                            )



                            st.write(

                                "Training kernel shape:",

                                load_training_kernel("700").shape

                            )



                            st.write(

                                "New kernel shape:",

                                K_new.shape

                            )



                            st.write(

                                "Raw OCSVM predictions:",

                                raw_predictions.tolist()

                            )



                            st.write(

                                "Decision scores:",

                                decision_scores.tolist()

                            )



                    except Exception as e:



                        st.error(

                            "Prediction failed."

                        )



                        st.exception(e)



        except Exception as e:



            st.error(

                "Could not read the uploaded CSV."

            )



            st.exception(e)





# ============================================================

# MODEL RESULTS

# ============================================================



elif page == "Model Results":



    st.header("Experimental Results")



    st.info(

        "Results are separated by training size. Each section compares "

        "the QML and classical One-Class SVM models using the same "

        "training-size experiment."

    )



    # ========================================================

    # 500-SAMPLE RESULTS

    # ========================================================



    st.subheader("500-Sample Experiment")



    st.caption(

        "QML: 500 normal training samples, ZZ, 4 qubits, reps=1, ν=0.15. "

        "Classical: 500 normal training samples, 4D PCA, RBF One-Class SVM, ν=0.15."

    )



    results_500 = pd.DataFrame({

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score",

            "ROC-AUC",

            "PR-AUC",

        ],

        "QML — 500 Samples": [

            85.20,

            55.92,

            78.70,

            65.38,

            84.66,

            55.55,

        ],

        "Classical — 500 Samples": [

            84.38,

            55.20,

            63.89,

            59.23,

            84.09,

            63.15,

        ],

    })



    st.dataframe(

        results_500,

        use_container_width=True,

        hide_index=True,

    )



    cm_col1, cm_col2 = st.columns(2)



    with cm_col1:

        st.metric("QML 500 Accuracy", "85.20%")

        st.write("Confusion Matrix")

        st.code(

            "[[433, 67],\n [ 23, 85]]",

            language="text",

        )



    with cm_col2:

        st.metric("Classical 500 Accuracy", "84.38%")

        st.write("Confusion Matrix")

        st.code(

            "[[444, 56],\n [ 39, 69]]",

            language="text",

        )



    # ========================================================

    # 700-SAMPLE RESULTS

    # ========================================================



    st.divider()

    st.subheader("700-Sample Experiment")



    st.caption(

        "QML: 700 normal training samples, ZZ, 4 qubits, reps=2, ν=0.08. "

        "Classical: 700 normal training samples, 4D PCA, RBF One-Class SVM, ν=0.08."

    )



    results_700 = pd.DataFrame({

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score",

            "ROC-AUC",

            "PR-AUC",

        ],

        "QML — 700 Samples": [

            86.68,

            62.39,

            62.96,

            62.67,

            88.21,

            66.13,

        ],

        "Classical — 700 Samples": [

            85.36,

            60.22,

            51.85,

            55.72,

            88.40,

            57.01,

        ],

    })



    st.dataframe(

        results_700,

        use_container_width=True,

        hide_index=True,

    )



    cm_col1, cm_col2 = st.columns(2)



    with cm_col1:

        st.metric("QML 700 Accuracy", "86.68%")

        st.write("Confusion Matrix")

        st.code(

            "[[459, 41],\n [ 40, 68]]",

            language="text",

        )



    with cm_col2:

        st.metric("Classical 700 Accuracy", "85.36%")

        st.write("Confusion Matrix")

        st.code(

            "[[463, 37],\n [ 52, 56]]",

            language="text",

        )



    st.divider()



    st.subheader("1000-Sample Experiment")
    st.caption(
        "1000 normal training samples • QML ZZ, 4 qubits, reps=2, ν=0.08 • "
        "Classical 4D PCA + RBF One-Class SVM, ν=0.08"
    )

    results_1000 = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
        "QML — 1000 Samples": [84.87, 59.52, 46.30, 52.08, 79.28, 46.96],
        "Classical — 1000 Samples": [84.87, 59.52, 46.30, 52.08, 84.99, 51.65],
    })
    st.dataframe(results_1000, use_container_width=True, hide_index=True)

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.metric("QML 1000 Accuracy", "84.87%")
        st.write("Confusion Matrix")
        st.code("[[466, 34],\n [ 58, 50]]", language="text")
    with cm_col2:
        st.metric("Classical 1000 Accuracy", "84.87%")
        st.write("Confusion Matrix")
        st.code("[[466, 34],\n [ 58, 50]]", language="text")

    st.info(
        "The 1000-sample experiment is a training-size study. It does not replace the "
        "700-sample active prototype. The frozen prototype remains the 700-sample QML model."
    )

    st.subheader("Training-Size Interpretation")



    st.write(

        "The 700-sample experiment increased the number of normal "

        "training transactions from 500 to 700. Accuracy increased "

        "for both QML and classical models. However, recall and F1 "

        "did not increase for either model, demonstrating that "

        "increasing training size does not guarantee improvement "

        "across every metric."

    )



    # ========================================================

    # REALISTIC PREVALENCE

    # ========================================================



    st.divider()

    st.subheader("Realistic-Prevalence Evaluation")



    _realistic_alt = os.path.join(
        RESULTS_DIR,
        "realistic",
        "realistic_comparison.csv"
    )

    realistic_path = (
        _realistic_alt
        if os.path.exists(_realistic_alt)
        else os.path.join(RESULTS_DIR, "realistic_comparison.csv")
    )



    if os.path.exists(realistic_path):



        realistic_df = pd.read_csv(

            realistic_path

        )



        display_df = realistic_df.copy()



        if "QML" in display_df.columns:

            display_df["QML"] = (

                display_df["QML"] * 100

            ).round(3)



        if "Classical" in display_df.columns:

            display_df["Classical"] = (

                display_df["Classical"] * 100

            ).round(3)



        st.dataframe(

            display_df,

            use_container_width=True

        )



        st.info(

            "This is a separate prevalence-matched evaluation from "

            "the earlier experiment using 5,780 normal and 10 fraud "

            "transactions. It is not a 700-sample result."

        )



    else:

        st.warning(

            "Realistic comparison file not found."

        )





# ============================================================

# GRAPHS

# ============================================================



elif page == "Graphs":



    st.header("Experimental Graphs")



    st.info(

        "All development graphs are separated by training size. "

        "The 500-sample and 700-sample experiments are shown independently. "

        "The realistic-prevalence evaluation remains a separate section."

    )



    # ========================================================

    # 500-SAMPLE GRAPHS

    # ========================================================



    st.subheader("500-Sample Experiment")



    st.caption(

        "500 normal training samples • QML ZZ reps=1 ν=0.15 • "

        "Classical RBF One-Class SVM ν=0.15"

    )



    metrics_500_graph = pd.DataFrame(

        {

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1 Score",

                "ROC-AUC",

                "PR-AUC",

            ],

            "QML": [

                85.20,

                55.92,

                78.70,

                65.38,

                84.66,

                55.55,

            ],

            "Classical": [

                84.38,

                55.20,

                63.89,

                59.23,

                84.09,

                63.15,

            ],

        }

    ).set_index("Metric")



    st.write("QML vs Classical — 500 Samples")

    st.bar_chart(

        metrics_500_graph,

        use_container_width=True,

    )



    st.write("ν Tuning — 500-Sample QML")



    nu_500 = pd.DataFrame(

        {

            "ν": [

                0.01, 0.03, 0.05,

                0.10, 0.15, 0.20

            ],

            "F1 Score": [

                56.00, 55.45, 57.43,

                56.54, 60.78, 60.14

            ],

            "ROC-AUC": [

                71.22, 71.40, 73.29,

                77.39, 81.87, 83.88

            ],

            "PR-AUC": [

                51.33, 51.46, 53.06,

                55.13, 56.23, 56.20

            ],

        }

    ).set_index("ν")



    st.line_chart(

        nu_500,

        use_container_width=True,

    )



    st.write("Feature-Map Tuning — 500-Sample QML")



    feature_map_500 = pd.DataFrame(

        {

            "Configuration": [

                "ZZ reps=1 linear",

                "ZZ reps=2 linear",

                "ZZ reps=2 full",

            ],

            "F1 Score": [

                65.38,

                60.78,

                58.65,

            ],

            "ROC-AUC": [

                84.66,

                81.87,

                79.63,

            ],

        }

    ).set_index("Configuration")



    st.bar_chart(

        feature_map_500,

        use_container_width=True,

    )



    # ========================================================

    # 700-SAMPLE GRAPHS

    # ========================================================



    st.divider()

    st.subheader("700-Sample Experiment")



    st.caption(

        "700 normal training samples • QML ZZ reps=2 ν=0.08 • "

        "Classical RBF One-Class SVM ν=0.08"

    )



    metrics_700_graph = pd.DataFrame(

        {

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1 Score",

                "ROC-AUC",

                "PR-AUC",

            ],

            "QML": [

                86.68,

                62.39,

                62.96,

                62.67,

                88.21,

                66.13,

            ],

            "Classical": [

                85.36,

                60.22,

                51.85,

                55.72,

                88.40,

                57.01,

            ],

        }

    ).set_index("Metric")



    st.write("QML vs Classical — 700 Samples")

    st.bar_chart(

        metrics_700_graph,

        use_container_width=True,

    )



    st.write("ν Tuning — 700-Sample QML")



    nu_700 = pd.DataFrame(

        {

            "ν": [

                0.01, 0.03, 0.05, 0.08,

                0.10, 0.12, 0.15, 0.20

            ],

            "F1 Score": [

                46.51, 44.97, 52.51, 60.87,

                64.04, 66.11, 66.67, 66.67

            ],

            "ROC-AUC": [

                80.36, 82.51, 84.33, 87.00,

                88.42, 89.14, 89.44, 89.59

            ],

            "PR-AUC": [

                54.07, 55.91, 56.02, 60.42,

                62.22, 63.26, 63.50, 63.76

            ],

        }

    ).set_index("ν")



    st.line_chart(

        nu_700,

        use_container_width=True,

    )



    st.write("Circuit Repetition Comparison — 700 Samples")



    reps_700 = pd.DataFrame(

        {

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1 Score",

                "ROC-AUC",

                "PR-AUC",

            ],

            "Reps=1": [

                86.68,

                63.64,

                58.33,

                60.87,

                87.00,

                60.42,

            ],

            "Reps=2": [

                86.68,

                62.39,

                62.96,

                62.67,

                88.21,

                66.13,

            ],

        }

    ).set_index("Metric")



    st.bar_chart(

        reps_700,

        use_container_width=True,

    )



    # ========================================================

                # 1000-SAMPLE GRAPHS
    # ========================================================

    st.divider()
    st.subheader("1000-Sample Experiment")
    st.caption(
    "1000 normal training samples • QML ZZ reps=2 ν=0.08 • "
    "Classical RBF One-Class SVM ν=0.08"
    )

    metrics_1000_graph = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"],
    "QML": [84.87, 59.52, 46.30, 52.08, 79.28, 46.96],
    "Classical": [84.87, 59.52, 46.30, 52.08, 84.99, 51.65],
    }).set_index("Metric")
    st.write("QML vs Classical — 1000 Samples")
    st.bar_chart(metrics_1000_graph, use_container_width=True)

    st.write("1000-Sample ν Sensitivity")
    nu_1000 = pd.DataFrame({
    "ν": [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
    "Accuracy": [82.89, 82.40, 83.55, 84.87, 85.03, 85.36, 85.36, 84.21, 82.40],
    "F1 Score": [25.71, 28.19, 40.48, 52.08, 55.61, 57.42, 60.44, 60.66, 58.69],
    "ROC-AUC": [66.46, 67.30, 72.45, 79.28, 81.71, 83.29, 84.01, 83.94, 83.94],
    "PR-AUC": [33.19, 37.16, 41.13, 46.96, 49.11, 50.20, 50.89, 50.39, 50.12],
    }).set_index("ν")
    st.line_chart(nu_1000, use_container_width=True)

# TRAINING-SIZE COMPARISON

    # ========================================================



    st.divider()

    st.subheader("Training-Size Comparison")



    size_comparison = pd.DataFrame(

        {

            "Metric": [

                "Accuracy",

                "Precision",

                "Recall",

                "F1 Score",

                "ROC-AUC",

                "PR-AUC",

            ],

            "QML — 500": [

                85.20, 55.92, 78.70,

                65.38, 84.66, 55.55

            ],

            "Classical — 500": [

                84.38, 55.20, 63.89,

                59.23, 84.09, 63.15

            ],

            "QML — 700": [

                86.68, 62.39, 62.96,

                62.67, 88.21, 66.13

            ],

            "Classical — 700": [

                85.36, 60.22, 51.85,

                55.72, 88.40, 57.01

            ],

            "QML — 1000": [
                84.87, 59.52, 46.30,
                52.08, 79.28, 46.96
            ],

            "Classical — 1000": [
                84.87, 59.52, 46.30,
                52.08, 84.99, 51.65
            ],

        }

    ).set_index("Metric")



    st.bar_chart(

        size_comparison,

        use_container_width=True,

    )



    # ========================================================

    # REALISTIC PREVALENCE

    # ========================================================



    st.divider()

    st.subheader("Realistic-Prevalence Evaluation")



    st.info(

        "This section is intentionally separate from the 500/700 "

        "development experiments. It uses the earlier prevalence-matched "

        "evaluation with 5,780 normal and 10 fraud transactions."

    )



    _realistic_alt = os.path.join(
        RESULTS_DIR,
        "realistic",
        "realistic_comparison.csv"
    )

    realistic_path = (
        _realistic_alt
        if os.path.exists(_realistic_alt)
        else os.path.join(RESULTS_DIR, "realistic_comparison.csv")
    )



    if os.path.exists(realistic_path):



        realistic_df = pd.read_csv(

            realistic_path

        )



        st.dataframe(

            realistic_df,

            use_container_width=True,

            hide_index=True,

        )



        st.caption(

            "Only the previously completed realistic-prevalence "

            "experiment is shown here; no 700-sample realistic "

            "evaluation is claimed."

        )



    else:



        st.warning(

            "Realistic comparison file not found."

        )





# ============================================================

# FOOTER

# ============================================================



st.divider()



st.caption(

    "Quantum-Kernel One-Class SVM • "

    "Credit Card Fraud Detection • "

    "500 / 700 / 1000 Training-Size Experiments • "

    "700-Sample Active Prototype"

)
