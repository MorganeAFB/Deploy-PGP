"""
features.py — engenharia de atributos compartilhada

Este módulo é a ÚNICA fonte da verdade sobre como um caso vira as colunas
que o modelo espera. O notebook de treino e o app Streamlit importam daqui.

Motivo: se a engenharia ficar escrita em dois lugares, treino e produção
divergem com o tempo e o modelo passa a prever errado sem dar erro.
É o problema conhecido como training-serving skew.
"""

import pandas as pd

# =============================================================================
# Grupos de campos da ficha do SINAN
# =============================================================================

VIOLENCE_TYPES = [
    "VIOL_FISIC", "VIOL_PSICO", "VIOL_NEGLI", "VIOL_FINAN",
    "VIOL_SEXU", "VIOL_TORT", "VIOL_TRAF", "VIOL_LEGAL",
]

AGGRESSION_MEANS = [
    "AG_FORCA", "AG_AMEACA", "AG_OBJETO", "AG_CORTE",
    "AG_QUENTE", "AG_ENFOR", "AG_ENVEN", "AG_FOGO",
]

FAMILY = [
    "REL_PAI", "REL_MAE", "REL_PAD", "REL_MAD",
    "REL_FILHO", "REL_IRMAO", "REL_CONJ", "REL_EXCON",
]

# Coabitação provável. Ex-cônjuge fica fora por definição.
HOUSEHOLD = ["REL_CONJ", "REL_FILHO", "REL_CUIDA"]

DISABILITY = [
    "DEF_FISICA", "DEF_MENTAL", "DEF_VISUAL", "DEF_AUDITI",
    "TRAN_MENT", "TRAN_COMP", "DEF_OUT",
]

AGE_BINS = [60, 70, 80, 90, 120]
AGE_LABELS = ["60-69", "70-79", "80-89", "90+"]

ENGINEERED = [
    "n_violence_types", "n_aggression_means", "family_aggressor",
    "household_aggressor", "aggressor_identified", "has_disability",
    "age_band",
]


# =============================================================================
# Normalização
# =============================================================================

def normalize_code(value):
    """Padroniza o código antes de comparar.

    O parquet traz '1', ' 1', 1 ou 1.0 para o mesmo valor. Sem padronizar,
    a comparação falha em silêncio.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text or None


def _is_yes(df, column):
    """True onde o campo Sim/Não está marcado como Sim (código 1)."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].map(normalize_code) == "1"


# =============================================================================
# Engenharia de atributos
# =============================================================================

def build_features(df):
    """Acrescenta as variáveis derivadas ao DataFrame.

    Recebe os campos crus do SINAN (códigos) mais a coluna `Idade` em anos.
    Devolve uma cópia com as colunas de ENGINEERED acrescentadas.

    Funciona tanto para a base inteira no treino quanto para uma linha só
    vinda do formulário — é exatamente esse o ponto.
    """
    out = df.copy()

    # Contagens: violência e meio múltiplos indicam gravidade
    out["n_violence_types"] = sum(
        _is_yes(out, c).astype(int) for c in VIOLENCE_TYPES
    )
    out["n_aggression_means"] = sum(
        _is_yes(out, c).astype(int) for c in AGGRESSION_MEANS
    )

    # Agregação do agressor: o que importa é a categoria, não qual parente
    relations = [c for c in out.columns if c.startswith("REL_")]

    out["family_aggressor"] = sum(
        _is_yes(out, c).astype(int) for c in FAMILY
    ) > 0
    out["household_aggressor"] = sum(
        _is_yes(out, c).astype(int) for c in HOUSEHOLD
    ) > 0
    out["aggressor_identified"] = sum(
        _is_yes(out, c).astype(int) for c in relations
    ) > 0

    out["has_disability"] = sum(
        _is_yes(out, c).astype(int) for c in DISABILITY
    ) > 0

    # Faixa etária: a relação com recorrência pode não ser linear
    out["age_band"] = pd.cut(
        out["Idade"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(str)

    return out


def align_to_model(df, pipeline):
    """Garante que o DataFrame tem exatamente as colunas que o modelo espera.

    Coluna faltando entra como nula (o imputer do pipeline resolve);
    coluna sobrando é descartada; a ordem é a do treino.

    Sem isso, um campo a mais ou a menos no formulário quebra a predição
    — ou, pior, desloca os valores sem dar erro.
    """
    expected = list(pipeline.named_steps["prep"].feature_names_in_)

    for column in expected:
        if column not in df.columns:
            df[column] = None

    return df[expected]
