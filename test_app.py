"""
test_app.py — testes da aplicação de triagem

Rodar:  pytest -v

Cobre os três níveis que a rubrica da Sprint 3 pede:
  - unitários    : engenharia de atributos, campo a campo
  - integração   : formulário -> features -> modelo -> probabilidade
  - aceite       : comportamento esperado do produto
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from features import (
    build_features, align_to_model, normalize_code,
    VIOLENCE_TYPES, AGGRESSION_MEANS, ENGINEERED,
)

MODEL_FILE = "modelo_risco_recorrencia.joblib"


# =============================================================================
# Fixtures
# =============================================================================

def make_case(**overrides):
    """Monta um caso completo com valores neutros, sobrescrevendo o que vier."""
    case = {"Idade": 75.0, "CS_SEXO": "F"}

    for field in VIOLENCE_TYPES + AGGRESSION_MEANS:
        case[field] = "2"

    for field in ["REL_FILHO", "REL_CONJ", "REL_EXCON", "REL_CUIDA",
                  "REL_PAI", "REL_MAE", "REL_IRMAO", "REL_CONHEC",
                  "REL_DESCO", "REL_INST", "REL_PATRAO", "REL_POL",
                  "REL_OUTROS"]:
        case[field] = "2"

    for field in ["DEF_FISICA", "DEF_MENTAL", "DEF_VISUAL", "DEF_AUDITI",
                  "TRAN_MENT", "TRAN_COMP", "DEF_OUT"]:
        case[field] = "2"

    case.update({
        "SIT_CONJUG": "9", "CS_ESCOL_N": "9", "LOCAL_OCOR": "01",
        "VIOL_MOTIV": "88", "AUTOR_SEXO": "9", "AUTOR_ALCO": "2",
        "NUM_ENVOLV": "9", "DEF_TRANS": "2",
    })

    case.update(overrides)
    return pd.DataFrame([case])


@pytest.fixture(scope="module")
def pipeline():
    if not Path(MODEL_FILE).exists():
        pytest.skip(f"{MODEL_FILE} ausente — rode o notebook de treino antes")
    return joblib.load(MODEL_FILE)


# =============================================================================
# Unitários — normalização
# =============================================================================

@pytest.mark.parametrize("entrada,esperado", [
    ("1", "1"),
    (" 1 ", "1"),      # espaço sobrando no parquet
    (1, "1"),          # veio como inteiro
    (1.0, "1"),        # veio como float
    ("1.0", "1"),
    ("", None),
    (None, None),
])
def test_normalize_code(entrada, esperado):
    assert normalize_code(entrada) == esperado


# =============================================================================
# Unitários — engenharia de atributos
# =============================================================================

def test_conta_tipos_de_violencia():
    caso = make_case(VIOL_FISIC="1", VIOL_PSICO="1", VIOL_NEGLI="1")
    resultado = build_features(caso)
    assert resultado["n_violence_types"].iloc[0] == 3


def test_ignorado_nao_conta_como_sim():
    """Código 9 é desconhecimento, não afirmação."""
    caso = make_case(VIOL_FISIC="9", VIOL_PSICO="9")
    resultado = build_features(caso)
    assert resultado["n_violence_types"].iloc[0] == 0


def test_agressor_familiar():
    caso = make_case(REL_FILHO="1")
    resultado = build_features(caso)
    assert resultado["family_aggressor"].iloc[0]
    assert resultado["household_aggressor"].iloc[0]
    assert resultado["aggressor_identified"].iloc[0]


def test_cuidador_coabita_mas_nao_e_familia():
    caso = make_case(REL_CUIDA="1")
    resultado = build_features(caso)
    assert not resultado["family_aggressor"].iloc[0]
    assert resultado["household_aggressor"].iloc[0]


def test_ex_conjuge_e_familia_mas_nao_coabita():
    caso = make_case(REL_EXCON="1")
    resultado = build_features(caso)
    assert resultado["family_aggressor"].iloc[0]
    assert not resultado["household_aggressor"].iloc[0]


def test_autor_nao_identificado():
    """13,9% da base cai neste caso — precisa virar flag, não erro."""
    resultado = build_features(make_case())
    assert not resultado["aggressor_identified"].iloc[0]


@pytest.mark.parametrize("idade,faixa", [
    (60, "60-69"), (69, "60-69"),
    (70, "70-79"),   # fronteira: right=False joga para a faixa de cima
    (85, "80-89"), (95, "90+"),
])
def test_faixa_etaria(idade, faixa):
    resultado = build_features(make_case(Idade=float(idade)))
    assert resultado["age_band"].iloc[0] == faixa


def test_todas_as_derivadas_sao_criadas():
    resultado = build_features(make_case())
    faltando = [c for c in ENGINEERED if c not in resultado.columns]
    assert not faltando, f"derivadas ausentes: {faltando}"


def test_build_features_nao_altera_a_entrada():
    """Efeito colateral em DataFrame é fonte clássica de bug silencioso."""
    caso = make_case()
    antes = list(caso.columns)
    build_features(caso)
    assert list(caso.columns) == antes


# =============================================================================
# Integração — alinhamento com o modelo
# =============================================================================

def test_alinhamento_devolve_as_colunas_do_treino(pipeline):
    esperadas = list(pipeline.named_steps["prep"].feature_names_in_)
    alinhado = align_to_model(build_features(make_case()), pipeline)
    assert list(alinhado.columns) == esperadas


def test_campo_faltando_nao_quebra(pipeline):
    """Formulário incompleto deve gerar nulo, não exceção."""
    caso = make_case().drop(columns=["AUTOR_ALCO"])
    alinhado = align_to_model(build_features(caso), pipeline)
    assert len(alinhado) == 1


def test_campo_extra_e_descartado(pipeline):
    caso = make_case(CAMPO_INVENTADO="x")
    alinhado = align_to_model(build_features(caso), pipeline)
    assert "CAMPO_INVENTADO" not in alinhado.columns


# =============================================================================
# Integração — predição de ponta a ponta
# =============================================================================

def test_predicao_devolve_probabilidade_valida(pipeline):
    alinhado = align_to_model(build_features(make_case()), pipeline)
    prob = pipeline.predict_proba(alinhado)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_predicao_e_deterministica(pipeline):
    """Mesmo caso, mesma resposta. Sem isso não há como auditar."""
    alinhado = align_to_model(build_features(make_case()), pipeline)
    a = pipeline.predict_proba(alinhado)[0, 1]
    b = pipeline.predict_proba(alinhado)[0, 1]
    assert a == b


def test_casos_diferentes_geram_respostas_diferentes(pipeline):
    """Se tudo der a mesma nota, a fila não é ordenada por nada."""
    leve = align_to_model(build_features(make_case()), pipeline)
    grave = align_to_model(
        build_features(make_case(
            VIOL_FISIC="1", VIOL_PSICO="1", VIOL_NEGLI="1",
            REL_FILHO="1", AUTOR_ALCO="1", AG_FORCA="1",
        )),
        pipeline,
    )
    assert pipeline.predict_proba(leve)[0, 1] != pipeline.predict_proba(grave)[0, 1]


def test_lote_e_individual_dao_o_mesmo_resultado(pipeline):
    """Prever 1 caso ou 10 de uma vez não pode mudar a nota de cada um."""
    casos = pd.concat([
        make_case(Idade=65.0),
        make_case(Idade=80.0, REL_FILHO="1"),
        make_case(Idade=92.0, VIOL_NEGLI="1"),
    ], ignore_index=True)

    lote = pipeline.predict_proba(align_to_model(build_features(casos), pipeline))[:, 1]

    for i in range(len(casos)):
        unico = casos.iloc[[i]].reset_index(drop=True)
        individual = pipeline.predict_proba(
            align_to_model(build_features(unico), pipeline)
        )[0, 1]
        assert abs(lote[i] - individual) < 1e-9


# =============================================================================
# Aceite — requisitos do produto
# =============================================================================

def test_modelo_carrega_do_arquivo():
    """Se falhar aqui, o app não sobe no Streamlit Cloud."""
    assert Path(MODEL_FILE).exists()
    assert joblib.load(MODEL_FILE) is not None


def test_metadados_declaram_limitacao():
    """Requisito ético: o aviso de uso acompanha o modelo."""
    meta_file = Path(MODEL_FILE.replace(".joblib", "_metadados.json"))
    if not meta_file.exists():
        pytest.skip("metadados ausentes")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert "aviso_uso" in meta
    assert len(meta["aviso_uso"]) > 20
