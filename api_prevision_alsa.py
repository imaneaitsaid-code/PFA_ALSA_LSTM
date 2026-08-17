"""
API FastAPI - Prevision de la demande voyageurs ALSA
=====================================================
Expose les modeles LSTM entraines dans le notebook Colab (PFA_ALSA_LSTM.ipynb) :
- un LSTM PASSAGERS par ligne (lstm_ligne_<NOM>.keras)
- un LSTM-BUS DEDIE par ligne (lstm_bus_ligne_<NOM>.keras), qui apprend directement
  la relation passagers/jour/saison -> nombre de bus a partir de l'historique reel de
  chaque ligne. Il n'y a plus de calcul par rotation/capacite : le nombre de bus est
  une prevision de modele, pas une formule.

Arborescence attendue a cote de ce fichier (a copier depuis Drive) :
    models/
        lstm_ligne_<NOM>.keras          scaler_ligne_<NOM>.pkl          (LSTM passagers)
        lstm_bus_ligne_<NOM>.keras      scaler_bus_ligne_<NOM>.pkl      (LSTM-bus dedie)
    data/
        BD_EXPLOITATION_clean.csv

Endpoints :
    GET    /prevision/{ligne}?date=2026-08-15                  prevision pour une date precise
    GET    /prevision/{ligne}?date_debut=...&date_fin=...      prevision sur un intervalle de dates
    POST   /donnees-quotidiennes                                le backend envoie ici les nouvelles
                                                                 donnees reelles du jour
    POST   /declarer-jour-aid                                   marque une date comme jour de l'Aid
                                                                 (ex. annee future non encore au calendrier)
    DELETE /declarer-jour-aid/{date}                            annule une declaration faite par erreur

Lancer en local : uvicorn api_prevision_alsa:app --reload --port 8000
Documentation interactive auto-generee : http://localhost:8000/docs
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------
# Chemins et constantes
# --------------------------------------------------------------------------
MODELS_DIR = "models"
DATA_PATH = "data/processed/BD_EXPLOITATION_clean.csv"
WINDOW_SIZE = 30

FEATURES = ['passagers', 'is_vacances', 'is_weekend', 'is_ferie', 'is_eid',
            'jour_0', 'jour_1', 'jour_2', 'jour_3', 'jour_4', 'jour_5', 'jour_6',
            'sin_mois', 'cos_mois',
            'lag_1', 'lag_7', 'lag_365', 'lag_730',
            'rolling_mean_7', 'rolling_mean_30']
FEATURES_BUS = FEATURES + ['bus']   # memes features + l'historique des bus (la cible)
IDX_BUS = len(FEATURES)             # position de la colonne 'bus' (ajoutee en dernier)

# --------------------------------------------------------------------------
# Calendrier des jours feries marocains (une SEULE definition - pas de doublon)
# A METTRE A JOUR CHAQUE ANNEE (fetes religieuses annoncees quelques jours a l'avance)
# --------------------------------------------------------------------------
DATES_AID = [
    '2024-04-10', '2024-04-11', '2025-03-31', '2025-04-01', '2026-03-20', '2026-03-21',
    '2024-06-17', '2024-06-18', '2025-06-06', '2025-06-07', '2026-05-27', '2026-05-28',
]
DATES_AUTRES_FERIES_RELIGIEUX = [
    '2024-07-07', '2024-09-16', '2025-06-27', '2025-09-05', '2026-06-17', '2026-08-26',
]
JOURS_FERIES_FIXES = [(1, 1), (1, 11), (1, 14), (5, 1), (7, 30), (8, 14), (8, 20), (8, 21), (11, 6), (11, 18)]
ANNEES_COUVERTES = [2024, 2025, 2026, 2027]
_dates_feries_fixes = [pd.Timestamp(annee, mois, jour)
                        for annee in ANNEES_COUVERTES for (mois, jour) in JOURS_FERIES_FIXES]
SET_DATES_AID = set(pd.to_datetime(DATES_AID))
SET_DATES_FERIES = set(pd.to_datetime(DATES_AUTRES_FERIES_RELIGIEUX)) | set(_dates_feries_fixes)

# Declaration manuelle : marquer un jour comme Aid meme s'il n'est pas (encore) dans le
# calendrier code en dur ci-dessus (annee future, ex. 2027, ou correction).
JOURS_AID_DECLARES = set()


def _est_ferie_civil(date):
    """Jours feries CIVILS uniquement (dates fixes + Moharram/Mawlid) : comportement
    assimile a un dimanche. L'Aid n'en fait PAS partie (voir _est_aid) : son comportement
    est bien plus marque qu'un simple dimanche (frequentation souvent < 50% de la normale)."""
    return date.normalize() in SET_DATES_FERIES


def _est_aid(date):
    """Aid El Fitr / Aid El Adha : comportement PROPRE, PAS assimile a un dimanche."""
    d = date.normalize()
    return (d in SET_DATES_AID) or (d in JOURS_AID_DECLARES)


def _est_ferie(date):
    """Jour ferie au sens large (civil OU Aid) - feature generale 'jour special'."""
    return _est_ferie_civil(date) or _est_aid(date)


# --------------------------------------------------------------------------
# App FastAPI
# --------------------------------------------------------------------------
app = FastAPI(title="API Prevision ALSA", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # a restreindre a l'URL du frontend en production
    allow_methods=["*"],
    allow_headers=["*"],
)

_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
_df["ligne"] = _df["ligne"].astype(str)

_model_cache = {}   # {ligne: (modele_p, scaler_p, modele_b, scaler_b)}


# --------------------------------------------------------------------------
# Schemas d'entree
# --------------------------------------------------------------------------
class DonneeJournaliere(BaseModel):
    """Une ligne de donnees reelles pour UNE ligne de bus, UN jour. C'est ce que le
    backend envoie chaque jour via POST /donnees-quotidiennes."""
    date: str = Field(..., description="Format YYYY-MM-DD", examples=["2026-08-15"])
    ligne: str = Field(..., examples=["12"])
    passagers: float = Field(..., ge=0)
    bus: float = Field(..., ge=0)
    horas: float = Field(..., ge=0, description="Heures cumulees de circulation ce jour-la")
    type_jour: str = Field(..., description="'LV', 'S', 'D' ou 'OTROS' (vacances scolaires)")
    type_ligne: str = Field(..., description="'URB' ou 'REG'")


class DeclarationJourAid(BaseModel):
    date: str = Field(..., description="Format YYYY-MM-DD", examples=["2027-03-10"])


# --------------------------------------------------------------------------
# Chargement des modeles (passagers + bus, mis en cache)
# --------------------------------------------------------------------------
def _nom_fichier(ligne):
    return str(ligne).replace(" ", "_").replace("/", "_")


def _charger_modeles(ligne):
    if ligne in _model_cache:
        return _model_cache[ligne]

    safe = _nom_fichier(ligne)
    chemin_modele_p = f"{MODELS_DIR}/lstm_ligne_{safe}.keras"
    chemin_scaler_p = f"{MODELS_DIR}/scaler_ligne_{safe}.pkl"
    chemin_modele_b = f"{MODELS_DIR}/lstm_bus_ligne_{safe}.keras"
    chemin_scaler_b = f"{MODELS_DIR}/scaler_bus_ligne_{safe}.pkl"

    if not all(os.path.exists(p) for p in [chemin_modele_p, chemin_scaler_p, chemin_modele_b, chemin_scaler_b]):
        raise HTTPException(404, f"Modele(s) manquant(s) (passagers et/ou bus) pour la ligne '{ligne}'")

    modele_p = load_model(chemin_modele_p)
    scaler_p = joblib.load(chemin_scaler_p)
    modele_b = load_model(chemin_modele_b)
    scaler_b = joblib.load(chemin_scaler_b)

    _model_cache[ligne] = (modele_p, scaler_p, modele_b, scaler_b)
    return _model_cache[ligne]


# --------------------------------------------------------------------------
# Preparation des donnees et features (identique au notebook)
# --------------------------------------------------------------------------
def _construire_historique(ligne):
    historique = _df[_df["ligne"] == ligne].sort_values("date").set_index("date")
    historique = historique[["passagers", "type_jour", "type_ligne", "bus", "horas"]].asfreq("D")
    historique["passagers"] = historique["passagers"].interpolate(method="time")
    historique["type_ligne"] = historique["type_ligne"].ffill().bfill()
    historique["type_jour"] = historique["type_jour"].fillna("LV")
    historique["bus"] = historique["bus"].interpolate(method="time").bfill().ffill()
    historique["horas"] = historique["horas"].interpolate(method="time").bfill().ffill()
    if str(ligne) == "73":
        historique = historique[~historique.index.month.isin([7, 8])]
    return historique


def _ajouter_features(d):
    d = d.copy()
    d["jour_semaine"] = d.index.dayofweek
    d["mois"] = d.index.month

    d["is_ferie"] = [int(_est_ferie_civil(dt)) for dt in d.index]
    d["is_eid"] = [int(_est_aid(dt)) for dt in d.index]
    d["is_vacances"] = (d["type_jour"] == "OTROS").astype(int)

    # jour ferie CIVIL -> encodage force sur "dimanche" ; l'Aid (is_eid) garde son vrai
    # jour de semaine, son comportement (bien plus marque) etant corrige a part
    jour_semaine_ajuste = d["jour_semaine"].where(d["is_ferie"] == 0, 6)
    d["is_weekend"] = np.where(d["is_ferie"] == 1, 1, (d["jour_semaine"] >= 5).astype(int))

    dummies_jour = pd.get_dummies(jour_semaine_ajuste, prefix="jour").astype(int)
    for i in range(7):
        col = f"jour_{i}"
        if col not in dummies_jour:
            dummies_jour[col] = 0
    d = pd.concat([d, dummies_jour], axis=1)

    d["sin_mois"] = np.sin(2 * np.pi * d["mois"] / 12)
    d["cos_mois"] = np.cos(2 * np.pi * d["mois"] / 12)

    d["lag_1"] = d["passagers"].shift(1)
    d["lag_7"] = d["passagers"].shift(7)
    d["rolling_mean_7"] = d["passagers"].shift(1).rolling(7).mean()
    d["rolling_mean_30"] = d["passagers"].shift(1).rolling(30).mean()
    d["lag_365"] = d["passagers"].shift(365).fillna(d["rolling_mean_30"])
    d["lag_730"] = d["passagers"].shift(730).fillna(d["lag_365"])

    # 'bus'/'horas' exclus du dropna : lors de la prevision recursive, les jours futurs
    # ajoutes n'ont pas encore ces valeurs connues au moment ou on construit la fenetre
    colonnes_essentielles = [c for c in d.columns if c not in ("bus", "horas")]
    return d.dropna(subset=colonnes_essentielles)


def _calculer_facteur_aid(feat_df):
    """Facteur multiplicatif median (frequentation Aid / frequentation normale) a partir
    des vrais jours d'Aid passes de cette ligne. None si aucun jour d'Aid dans l'historique."""
    jours_aid = feat_df[feat_df["is_eid"] == 1]
    if len(jours_aid) == 0:
        return None
    ratio = (jours_aid["passagers"] / jours_aid["rolling_mean_30"].replace(0, np.nan)).dropna()
    return float(ratio.median()) if len(ratio) else None


def _inverse_target(scaled_values, scaler, n_features, target_idx=0):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]


# --------------------------------------------------------------------------
# Post-traitement du nombre de bus
# --------------------------------------------------------------------------
def _arrondir_au_demi(x):
    return np.round(x * 2) / 2


def _post_traiter_bus(bus_brut, nb_passagers):
    """Arrondit a un multiple de 0.5, minimum 1 des qu'il y a du trafic. Si 0 passager
    (Aid extreme ou ligne 73 l'ete), comportement DIFFERENT : on force 0 bus directement,
    ce cas n'ayant jamais ete vu tel quel a l'entrainement du LSTM-bus."""
    if nb_passagers <= 0:
        return 0.0
    return max(_arrondir_au_demi(bus_brut), 1.0)


# --------------------------------------------------------------------------
# Prevision recursive (passagers puis bus), jusqu'a une date cible
# --------------------------------------------------------------------------
def _previsions_recursives(ligne, date_fin_cible):
    modele_p, scaler_p, modele_b, scaler_b = _charger_modeles(ligne)
    historique = _construire_historique(ligne)
    type_ligne = historique["type_ligne"].mode()[0]

    derniere_date_connue = historique.index[-1]
    horizon = (date_fin_cible - derniere_date_connue).days
    if horizon < 1:
        raise HTTPException(
            400,
            f"La date doit etre posterieure aux donnees disponibles (derniere date connue : "
            f"{derniere_date_connue.strftime('%Y-%m-%d')})."
        )
    if horizon > 60:
        raise HTTPException(
            400,
            "Date trop eloignee (plus de 60 jours) : la prevision recursive devient peu fiable "
            "au-dela de cet horizon."
        )

    facteur_aid = _calculer_facteur_aid(_ajouter_features(historique))

    chemin = []
    for _ in range(horizon):
        # IMPORTANT : on construit la fenetre AVANT d'ajouter le jour a predire.
        # Les deux modeles (passagers, bus) utilisent la MEME fenetre (30 jours
        # precedents, jamais le jour cible lui-meme) -> jamais de NaN, coherent avec
        # la facon dont les deux modeles ont ete entraines.
        feat = _ajouter_features(historique)
        prochaine_date = historique.index[-1] + timedelta(days=1)

        # --- etape 1 : prevision des passagers ---
        fenetre_p = feat[FEATURES].iloc[-WINDOW_SIZE:]
        Xp = scaler_p.transform(fenetre_p)[np.newaxis, :, :]
        yp_scaled = modele_p.predict(Xp, verbose=0).flatten()[0]
        p = float(_inverse_target(np.array([yp_scaled]), scaler_p, len(FEATURES))[0])

        if str(ligne) == "73" and prochaine_date.month in (7, 8):
            p = 0.0
        elif _est_aid(prochaine_date) and facteur_aid is not None:
            baseline_normale = feat["rolling_mean_30"].iloc[-1]
            p = baseline_normale * facteur_aid

        # un nombre de passagers est un entier positif : jamais negatif, jamais a virgule
        p = max(0, round(p))

        # --- etape 2 : prevision du nombre de bus (meme fenetre 'feat', pas de NaN) ---
        fenetre_b = feat[FEATURES_BUS].iloc[-WINDOW_SIZE:]
        Xb = scaler_b.transform(fenetre_b)[np.newaxis, :, :]
        yb_scaled = modele_b.predict(Xb, verbose=0).flatten()[0]
        b_brut = _inverse_target(np.array([yb_scaled]), scaler_b, len(FEATURES_BUS), target_idx=IDX_BUS)[0]
        b = 0.0 if str(ligne) == "73" and prochaine_date.month in (7, 8) else _post_traiter_bus(b_brut, p)

        chemin.append({
            "date": prochaine_date.strftime("%Y-%m-%d"),
            "passagers_predits": int(p),
            "bus_recommandes": b,
        })

        # on ajoute la ligne COMPLETE (passagers + bus) en une seule fois, une fois les
        # deux predictions faites - jamais de NaN insere dans l'historique
        historique.loc[prochaine_date] = {
            "passagers": p, "bus": b, "type_jour": "LV", "type_ligne": type_ligne,
        }

    return type_ligne, chemin


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.post("/donnees-quotidiennes")
def ajouter_donnees_quotidiennes(donnees: List[DonneeJournaliere]):
    """Ajoute (ou met a jour si la date+ligne existe deja) les donnees reelles du jour,
    envoyees par le backend. Pas de re-nettoyage complet : ces lignes sont deja propres,
    on les insere directement dans le jeu de donnees utilise pour les previsions."""
    global _df

    nouvelles = pd.DataFrame([d.model_dump() for d in donnees])
    nouvelles["date"] = pd.to_datetime(nouvelles["date"])
    nouvelles["ligne"] = nouvelles["ligne"].astype(str)

    cle_nouvelles = set(zip(nouvelles["date"], nouvelles["ligne"]))
    _df = _df[~_df.apply(lambda r: (r["date"], r["ligne"]) in cle_nouvelles, axis=1)]
    _df = pd.concat([_df, nouvelles], ignore_index=True).sort_values(["ligne", "date"])

    _df.to_csv(DATA_PATH, index=False)   # persiste sur disque

    lignes_touchees = nouvelles["ligne"].unique()
    for ligne in lignes_touchees:
        _model_cache.pop(ligne, None)   # invalide le cache modele pour forcer une relecture propre si besoin

    return {
        "statut": "ok",
        "nb_lignes_recues": len(nouvelles),
        "derniere_date_connue_par_ligne": {
            ligne: str(_df[_df["ligne"] == ligne]["date"].max().date())
            for ligne in lignes_touchees
        },
    }


@app.post("/declarer-jour-aid")
def declarer_jour_aid(declaration: DeclarationJourAid):
    """Marque une date comme jour de l'Aid, meme si elle n'est pas (encore) dans le
    calendrier code en dur (ex. annee future 2027+, dates annoncees quelques jours a l'avance)."""
    d = pd.Timestamp(declaration.date).normalize()
    JOURS_AID_DECLARES.add(d)
    return {"statut": "ok", "date_declaree": str(d.date()), "nb_jours_aid_declares_manuellement": len(JOURS_AID_DECLARES)}


@app.delete("/declarer-jour-aid/{date}")
def annuler_declaration_jour_aid(date: str):
    """Retire une date declaree par erreur."""
    d = pd.Timestamp(date).normalize()
    JOURS_AID_DECLARES.discard(d)
    return {"statut": "ok", "date_retiree": str(d.date())}


@app.get("/prevision/{ligne}")
def prevoir(ligne: str, date: str = None, date_debut: str = None, date_fin: str = None):
    """Prevision de passagers + bus recommandes.
    - Une date precise :      /prevision/12?date=2026-08-15
    - Un intervalle de dates : /prevision/12?date_debut=2026-08-10&date_fin=2026-08-20
    """
    if date and (date_debut or date_fin):
        raise HTTPException(400, "Precisez soit 'date', soit 'date_debut'+'date_fin', pas les deux.")
    if not date and not (date_debut and date_fin):
        raise HTTPException(400, "Precisez soit 'date', soit 'date_debut' ET 'date_fin'.")

    try:
        if date:
            date_debut_cible = date_fin_cible = pd.Timestamp(date)
        else:
            date_debut_cible = pd.Timestamp(date_debut)
            date_fin_cible = pd.Timestamp(date_fin)
    except ValueError:
        raise HTTPException(400, "date invalide, format attendu : YYYY-MM-DD")

    if date_fin_cible < date_debut_cible:
        raise HTTPException(400, "date_fin doit etre posterieure ou egale a date_debut")

    type_ligne, chemin_complet = _previsions_recursives(ligne, date_fin_cible)

    previsions_demandees = [
        p for p in chemin_complet if pd.Timestamp(p["date"]) >= date_debut_cible
    ]

    return {
        "ligne": ligne,
        "type_ligne": type_ligne,
        "date_debut": date_debut_cible.strftime("%Y-%m-%d"),
        "date_fin": date_fin_cible.strftime("%Y-%m-%d"),
        "previsions": previsions_demandees,
    }