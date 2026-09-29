"""
app.py — MVP de triagem de risco de recorrência

Recebe os dados de uma notificação, estima a probabilidade de a violência
ser de repetição e mostra quais características puxaram a pontuação.

Rodar local:   streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import build_features, align_to_model

MODEL_FILE = "modelo_risco_recorrencia.joblib"
META_FILE = "modelo_risco_recorrencia_metadados.json"

st.set_page_config(
    page_title="Triagem de risco — violência contra a pessoa idosa",
    layout="wide",
)


# =============================================================================
# Carga do modelo
# =============================================================================

@st.cache_resource
def load_model():
    """Carrega o pipeline uma vez e mantém em cache entre execuções."""
    if not Path(MODEL_FILE).exists():
        return None, None

    pipeline = joblib.load(MODEL_FILE)

    metadata = {}
    if Path(META_FILE).exists():
        metadata = json.loads(Path(META_FILE).read_text(encoding="utf-8"))

    return pipeline, metadata


pipeline, metadata = load_model()

if pipeline is None:
    st.error(
        f"Modelo não encontrado. Coloque `{MODEL_FILE}` na mesma pasta do app."
    )
    st.stop()


# =============================================================================
# Explicação por caso
# =============================================================================

REFERENCE = {
    # Valor de referência de cada campo: "Não" ou "Ignorado", conforme a ficha.
    # A explicação compara o caso real contra este cenário neutro.
    "default_yes_no": "2",
    "default_categorical": "9",
}


def explain(pipeline, row, fields, top=5):
    """Mede o quanto cada campo puxou a probabilidade para cima.

    Para cada campo preenchido, recalcula a probabilidade com aquele campo
    trocado pelo valor de referência ("Não"). A diferença é a contribuição
    daquele campo neste caso específico.

    Abordagem agnóstica ao modelo: funciona igual para regressão logística
    e para gradient boosting, ao contrário de ler coeficientes.
    """
    base_prob = float(pipeline.predict_proba(row)[0, 1])
    contributions = []

    for field, label in fields.items():
        if field not in row.columns:
            continue

        current = row[field].iloc[0]
        if current is None or str(current) == REFERENCE["default_yes_no"]:
            continue

        # Cenário contrafactual: e se este campo fosse "Não"?
        counterfactual = row.copy()
        counterfactual[field] = REFERENCE["default_yes_no"]
        counterfactual = build_features(counterfactual)
        counterfactual = align_to_model(counterfactual, pipeline)

        alternative = float(pipeline.predict_proba(counterfactual)[0, 1])
        delta = base_prob - alternative

        if abs(delta) > 0.001:
            contributions.append({"fator": label, "efeito": delta})

    contributions.sort(key=lambda item: item["efeito"], reverse=True)
    return base_prob, contributions[:top]


# =============================================================================
# Formulário
# =============================================================================

YES_NO = {"Não": "2", "Sim": "1", "Ignorado": "9"}

VIOLENCE_LABELS = {
    "VIOL_FISIC": "Violência física",
    "VIOL_PSICO": "Violência psicológica/moral",
    "VIOL_NEGLI": "Negligência/abandono",
    "VIOL_FINAN": "Violência financeira/econômica",
    "VIOL_SEXU": "Violência sexual",
    "VIOL_TORT": "Tortura",
    "VIOL_TRAF": "Tráfico de seres humanos",
    "VIOL_LEGAL": "Intervenção legal",
}

MEANS_LABELS = {
    "AG_FORCA": "Força corporal/espancamento",
    "AG_AMEACA": "Ameaça",
    "AG_OBJETO": "Objeto contundente",
    "AG_CORTE": "Objeto perfurocortante",
    "AG_QUENTE": "Substância/objeto quente",
    "AG_ENFOR": "Enforcamento",
    "AG_ENVEN": "Envenenamento/intoxicação",
    "AG_FOGO": "Arma de fogo",
}

AUTHOR_LABELS = {
    "REL_FILHO": "Filho(a)",
    "REL_CONJ": "Cônjuge",
    "REL_EXCON": "Ex-cônjuge",
    "REL_CUIDA": "Cuidador",
    "REL_PAI": "Pai",
    "REL_MAE": "Mãe",
    "REL_IRMAO": "Irmão(ã)",
    "REL_CONHEC": "Amigo/conhecido",
    "REL_DESCO": "Desconhecido",
    "REL_INST": "Relação institucional",
    "REL_PATRAO": "Patrão/chefe",
    "REL_POL": "Policial/agente da lei",
    "REL_OUTROS": "Outros",
}

DISABILITY_LABELS = {
    "DEF_FISICA": "Deficiência física",
    "DEF_MENTAL": "Deficiência intelectual",
    "DEF_VISUAL": "Deficiência visual",
    "DEF_AUDITI": "Deficiência auditiva",
    "TRAN_MENT": "Transtorno mental",
    "TRAN_COMP": "Transtorno de comportamento",
    "DEF_OUT": "Outras deficiências/síndromes",
}

LOCATIONS = {
    "Residência": "01", "Habitação coletiva": "02", "Escola": "03",
    "Local de prática esportiva": "04", "Bar ou similar": "05",
    "Via pública": "06", "Comércio/serviços": "07",
    "Indústrias/construção": "08", "Outro": "09", "Ignorado": "99",
}

MARITAL = {
    "Solteiro": "1", "Casado/união consensual": "2", "Viúvo": "3",
    "Separado": "4", "Ignorado": "9",
}

EDUCATION = {
    "Analfabeto": "0",
    "1ª a 4ª série incompleta": "1",
    "4ª série completa": "2",
    "5ª a 8ª série incompleta": "3",
    "Ensino fundamental completo": "4",
    "Ensino médio incompleto": "5",
    "Ensino médio completo": "6",
    "Educação superior incompleta": "7",
    "Educação superior completa": "8",
    "Ignorado": "9",
}

MOTIVATION = {
    "Não se aplica": "88", "Conflito geracional": "06",
    "Deficiência": "08", "Sexismo": "01", "Racismo": "03",
    "Situação de rua": "07", "Outros": "09", "Ignorado": "99",
}


st.title("Triagem de risco de recorrência")
st.caption(
    "Violência contra a pessoa idosa — apoio à priorização de casos notificados"
)

st.warning(
    "**Apoio à decisão humana.** A pontuação sugere uma ordem de "
    "atendimento; não substitui a avaliação profissional e não exclui "
    "nenhum caso da fila.",
    icon="⚠️",
)

with st.form("caso"):
    st.subheader("Perfil da vítima")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Idade", min_value=60, max_value=110, value=75)
    with col2:
        sex = st.selectbox("Sexo", ["Feminino", "Masculino", "Ignorado"])
    with col3:
        marital = st.selectbox("Situação conjugal", list(MARITAL))

    col4, col5 = st.columns(2)
    with col4:
        education = st.selectbox("Escolaridade", list(EDUCATION), index=9)
    with col5:
        location = st.selectbox("Local de ocorrência", list(LOCATIONS))

    st.subheader("Tipo de violência")
    st.caption("Marque todos os que se aplicam ao caso.")
    violence = st.multiselect(
        "Tipos", list(VIOLENCE_LABELS.values()), label_visibility="collapsed"
    )

    st.subheader("Meio de agressão")
    means = st.multiselect(
        "Meios", list(MEANS_LABELS.values()), label_visibility="collapsed"
    )

    st.subheader("Provável autor")
    authors = st.multiselect(
        "Autores", list(AUTHOR_LABELS.values()), label_visibility="collapsed"
    )

    col6, col7, col8 = st.columns(3)
    with col6:
        author_sex = st.selectbox(
            "Sexo do autor", ["Masculino", "Feminino", "Ambos", "Ignorado"]
        )
    with col7:
        alcohol = st.selectbox("Suspeita de álcool pelo autor", list(YES_NO))
    with col8:
        involved = st.selectbox(
            "Número de envolvidos", ["Um", "Dois ou mais", "Ignorado"]
        )

    st.subheader("Deficiência ou transtorno")
    disabilities = st.multiselect(
        "Condições", list(DISABILITY_LABELS.values()),
        label_visibility="collapsed",
    )

    motivation = st.selectbox("Violência motivada por", list(MOTIVATION))

    submitted = st.form_submit_button("Calcular risco", type="primary")


# =============================================================================
# Predição
# =============================================================================

if submitted:
    case = {
        "Idade": float(age),
        "CS_SEXO": {"Feminino": "F", "Masculino": "M", "Ignorado": "I"}[sex],
        "SIT_CONJUG": MARITAL[marital],
        "CS_ESCOL_N": EDUCATION[education],
        "LOCAL_OCOR": LOCATIONS[location],
        "VIOL_MOTIV": MOTIVATION[motivation],
        "AUTOR_SEXO": {
            "Masculino": "1", "Feminino": "2", "Ambos": "3", "Ignorado": "9",
        }[author_sex],
        "AUTOR_ALCO": YES_NO[alcohol],
        "NUM_ENVOLV": {"Um": "1", "Dois ou mais": "2", "Ignorado": "9"}[involved],
        "DEF_TRANS": "1" if disabilities else "2",
    }

    # Campos Sim/Não: marcado no formulário vira "1", o resto vira "2"
    for group, labels in [
        (violence, VIOLENCE_LABELS),
        (means, MEANS_LABELS),
        (authors, AUTHOR_LABELS),
        (disabilities, DISABILITY_LABELS),
    ]:
        for field, label in labels.items():
            case[field] = "1" if label in group else "2"

    row = pd.DataFrame([case])
    row = build_features(row)
    row = align_to_model(row, pipeline)

    all_labels = {
        **VIOLENCE_LABELS, **MEANS_LABELS,
        **AUTHOR_LABELS, **DISABILITY_LABELS,
        "AUTOR_ALCO": "Suspeita de uso de álcool",
    }
    probability, factors = explain(pipeline, row, all_labels)

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.metric("Probabilidade de recorrência", f"{probability:.0%}")

        if probability >= 0.70:
            st.error("Prioridade alta", icon="🔴")
        elif probability >= 0.50:
            st.warning("Prioridade média", icon="🟡")
        else:
            st.info("Prioridade padrão", icon="🔵")

        st.caption(
            "A prioridade reordena a fila. Nenhum caso deixa de ser atendido."
        )

    with right:
        st.subheader("O que pesou neste caso")

        if factors:
            chart = pd.DataFrame(factors).set_index("fator")
            st.bar_chart(chart, horizontal=True)
            st.caption(
                "Efeito estimado de cada característica sobre a probabilidade. "
                "Valores positivos elevam o risco; negativos reduzem."
            )
        else:
            st.info(
                "Nenhuma característica isolada alterou a estimativa de forma "
                "relevante neste caso."
            )

        st.caption(
            "Discorda? A avaliação profissional prevalece sobre a sugestão."
        )


# =============================================================================
# Rodapé: transparência sobre o modelo
# =============================================================================

with st.sidebar:
    st.header("Sobre o modelo")

    if metadata:
        st.write(f"**Algoritmo:** {metadata.get('modelo', '—')}")
        st.write(f"**Base:** {metadata.get('base', '—')}")
        st.write(f"**Treino:** {metadata.get('n_treino', '—'):,} casos")
        st.write(f"**Data:** {metadata.get('data_treino', '—')}")

        metrics = metadata.get("metricas", {})
        if metrics:
            st.metric("AUC no teste", metrics.get("AUC", "—"))
            st.caption(
                "AUC mede a capacidade de ordenar casos por risco. "
                "0,50 equivale ao acaso."
            )

    st.divider()
    st.subheader("Limitações")
    st.caption(
        "O modelo foi treinado apenas com casos notificados. Perfis "
        "sub-representados na base — como pessoas idosas isoladas ou com "
        "pouco acesso a serviços — tendem a receber pontuação mais baixa "
        "por falta de exemplos semelhantes no treino.\n\n"
        "O sistema prioriza casos já registrados; não identifica violência "
        "não notificada.\n\n"
        "A recorrência é usada como variável substituta de risco, por "
        "indisponibilidade de medida direta de gravidade nos microdados."
    )
