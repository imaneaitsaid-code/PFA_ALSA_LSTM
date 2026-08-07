"""
API FastAPI - Prevision de la demande voyageurs ALSA
=====================================================
Expose les modeles LSTM entraines dans le notebook Colab (PFA_ALSA_LSTM.ipynb).

Arborescence attendue a cote de ce fichier (a copier depuis Drive) :
    models/
        lstm_ligne_<NOM>.keras
        scaler_ligne_<NOM>.pkl
    data/
        BD_EXPLOITATION_clean.csv
    reports/
        synthese_metriques_par_ligne.csv   (contient rotations_par_jour par ligne)

Endpoints :
    GET  /lignes                                            liste des lignes + leurs metriques
    GET  /prevision/{ligne}?date=2026-08-15                  prevision pour une date precise
    GET  /prevision/{ligne}?date_debut=...&date_fin=...      prevision sur un intervalle de dates
    GET  /historique/{ligne}                                 donnees reelles passees (tout l'historique)
    GET  /historique/{ligne}?date_debut=...&date_fin=...     donnees reelles passees, filtrees
    POST /donnees-quotidiennes                                le backend envoie ici les nouvelles
                                                                donnees reelles du jour (voir plus bas)

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

MODELS_DIR = "models"
DATA_PATH = "data/processed/BD_EXPLOITATION_clean.csv"
SYNTHESE_PATH = "reports/synthese_metriques_par_ligne.csv"
WINDOW_SIZE = 14

FEATURES = ['passagers', 'is_vacances', 'is_weekend', 'is_ferie', 'is_eid',
            'jour_0', 'jour_1', 'jour_2', 'jour_3', 'jour_4', 'jour_5', 'jour_6',
            'sin_mois', 'cos_mois',
            'lag_1', 'lag_7', 'lag_365', 'lag_730',
            'rolling_mean_7', 'rolling_mean_30']

# Calendrier des jours feries marocains - DOIT rester identique a celui du notebook d'entrainement
JOURS_FERIES_FIXES = [(1, 1), (1, 11), (5, 1), (7, 30), (8, 14), (8, 20), (8, 21), (11, 6), (11, 18)]
DATES_AID_FITR = ['2024-04-10', '2025-03-31', '2026-03-20']
DATES_AID_ADHA = ['2024-06-17', '2025-06-07', '2025-06-09', '2026-05-27', '2026-05-28']
_DATES_AID = pd.to_datetime(DATES_AID_FITR).union(pd.to_datetime(DATES_AID_ADHA))


def _est_ferie(date):
    if (date.month, date.day) in JOURS_FERIES_FIXES:
        return True
    return date.normalize() in _DATES_AID


def _est_aid(date):
    return date.normalize() in _DATES_AID

CAPACITE_PAR_ERE = {
    "avant_2026": {"URB": 100, "REG": 90},
    "2026_et_plus": {"URB": 90, "REG": 75},
}

app = FastAPI(title="API Prevision ALSA", version="1.0")

# Autorise le frontend (autre origine/port) a appeler cette API depuis le navigateur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # a restreindre a l'URL du frontend en production
    allow_methods=["*"],
    allow_headers=["*"],
)

_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
_df["ligne"] = _df["ligne"].astype(str)
_synthese = pd.read_csv(SYNTHESE_PATH) if os.path.exists(SYNTHESE_PATH) else None

_model_cache = {}   # evite de recharger le modele a chaque requete


class DonneeJournaliere(BaseModel):
    """Une ligne de donnees reelles pour UNE ligne de bus, UN jour. C'est ce que le backend\n    envoie chaque jour via POST /donnees-quotidiennes."""
    date: str = Field(..., description="Format YYYY-MM-DD", examples=["2026-08-15"])
    ligne: str = Field(..., examples=["12"])
    passagers: float = Field(..., ge=0)
    bus: float = Field(..., ge=0)
    horas: float = Field(..., ge=0, description="Heures cumulees de circulation ce jour-la")
    type_jour: str = Field(..., description="'LV' (lundi-vendredi), 'S' (samedi), 'D' (dimanche), 'OTROS' (vacances scolaires)")
    type_ligne: str = Field(..., description="'URB' ou 'REG'")


@app.post("/donnees-quotidiennes")
def ajouter_donnees_quotidiennes(donnees: List[DonneeJournaliere]):
    """Ajoute (ou met a jour si la date+ligne existe deja) les donnees reelles du jour,
    envoyees par le backend. PAS de re-nettoyage complet ici (le nettoyage du fichier Excel
    brut, avec ses multiples onglets, ne se fait qu'une seule fois via le notebook) : on se
    contente d'inserer ces lignes deja propres directement dans le jeu de donnees utilise
    pour les previsions."""
    global _df

    nouvelles = pd.DataFrame([d.model_dump() for d in donnees])
    nouvelles["date"] = pd.to_datetime(nouvelles["date"])
    nouvelles["ligne"] = nouvelles["ligne"].astype(str)

    # on remplace les eventuels doublons (meme date+ligne deja presente), sinon on ajoute
    cle_nouvelles = set(zip(nouvelles["date"], nouvelles["ligne"]))
    _df = _df[~_df.apply(lambda r: (r["date"], r["ligne"]) in cle_nouvelles, axis=1)]
    _df = pd.concat([_df, nouvelles], ignore_index=True).sort_values(["ligne", "date"])

    _df.to_csv(DATA_PATH, index=False)   # persiste sur disque, pour survivre a un redemarrage de l'API

    return {
        "statut": "ok",
        "nb_lignes_recues": len(nouvelles),
        "derniere_date_connue_par_ligne": {
            ligne: str(_df[_df["ligne"] == ligne]["date"].max().date())
            for ligne in nouvelles["ligne"].unique()
        },
    }


def _nom_fichier(ligne):
    return str(ligne).replace(" ", "_").replace("/", "_")


def _charger_modele(ligne):
    if ligne in _model_cache:
        return _model_cache[ligne]
    chemin_modele = f"{MODELS_DIR}/lstm_ligne_{_nom_fichier(ligne)}.keras"
    chemin_scaler = f"{MODELS_DIR}/scaler_ligne_{_nom_fichier(ligne)}.pkl"
    if not (os.path.exists(chemin_modele) and os.path.exists(chemin_scaler)):
        raise HTTPException(404, f"Aucun modele entraine trouve pour la ligne '{ligne}'")
    modele = load_model(chemin_modele)
    scaler = joblib.load(chemin_scaler)
    _model_cache[ligne] = (modele, scaler)
    return modele, scaler


def _ajouter_features(d):
    d = d.copy()
    d["jour_semaine"] = d.index.dayofweek
    d["mois"] = d.index.month
    d["is_ferie"] = [int(_est_ferie(dt)) for dt in d.index]
    d["is_eid"] = [int(_est_aid(dt)) for dt in d.index]
    d["is_vacances"] = (d["type_jour"] == "OTROS").astype(int)
    # un jour ferie (JF fixe ou Aid) se comporte comme un dimanche, quel que soit le vrai jour
    # de la semaine -> on aligne l'encodage jour_semaine/is_weekend sur dimanche pour ces dates
    jour_semaine_ajuste = d["jour_semaine"].where(d["is_ferie"] == 0, 6)
    d["is_weekend"] = np.where(d["is_ferie"] == 1, 1, (d["jour_semaine"] >= 5).astype(int))
    dummies = pd.get_dummies(jour_semaine_ajuste, prefix="jour").astype(int)
    for i in range(7):
        col = f"jour_{i}"
        d[col] = dummies[col] if col in dummies else 0
    d["sin_mois"] = np.sin(2 * np.pi * d["mois"] / 12)
    d["cos_mois"] = np.cos(2 * np.pi * d["mois"] / 12)
    d["lag_1"] = d["passagers"].shift(1)
    d["lag_7"] = d["passagers"].shift(7)
    d["rolling_mean_7"] = d["passagers"].shift(1).rolling(7).mean()
    d["rolling_mean_30"] = d["passagers"].shift(1).rolling(30).mean()
    d["lag_365"] = d["passagers"].shift(365).fillna(d["rolling_mean_30"])
    d["lag_730"] = d["passagers"].shift(730).fillna(d["lag_365"])
    return d


def _capacite_du_jour(date, type_ligne):
    era = "avant_2026" if date.year < 2026 else "2026_et_plus"
    return CAPACITE_PAR_ERE[era][type_ligne]


def _bus_necessaires(nb_passagers, date, type_ligne, rotations_par_jour):
    if nb_passagers <= 0:
        return 0.0
    capacite = _capacite_du_jour(date, type_ligne) * rotations_par_jour
    bus_arrondi = round((nb_passagers / capacite) * 2) / 2
    # meme une ligne a faible frequentation tourne toute la journee (matin ET soir) :
    # des qu'il y a du trafic, le minimum realiste est 1 bus (journee complete), pas 0.5.
    return max(bus_arrondi, 1.0)


@app.get("/lignes")
def lister_lignes():
    """Liste des lignes disponibles avec leurs metriques (issues de la synthese du notebook)."""
    if _synthese is None:
        return {"lignes": sorted(_df["ligne"].unique().tolist())}
    return _synthese.to_dict(orient="records")


def _construire_historique(ligne):
    historique = _df[_df["ligne"] == ligne].sort_values("date").set_index("date")
    historique = historique[["passagers", "type_jour", "type_ligne", "bus"]].asfreq("D")
    historique["passagers"] = historique["passagers"].interpolate(method="time")
    historique["type_ligne"] = historique["type_ligne"].ffill().bfill()
    historique["type_jour"] = historique["type_jour"].fillna("LV")
    if str(ligne) == "73":
        historique = historique[~historique.index.month.isin([7, 8])]
    return historique


def _rotations_par_jour(ligne):
    if _synthese is not None:
        ligne_row = _synthese[_synthese["ligne"] == ligne]
        if len(ligne_row):
            return float(ligne_row["rotations_par_jour"].iloc[0])
    return 8.0  # valeur de repli si la synthese n'est pas fournie


def _previsions_recursives(ligne, date_fin_cible):
    """Prevoit jour par jour depuis la derniere donnee disponible jusqu'a date_fin_cible incluse.
    Renvoie la liste complete des jours prevus (necessaire en interne pour reconstruire les lags,
    meme si l'appelant ne veut afficher qu'une partie de cette liste)."""
    modele, scaler = _charger_modele(ligne)
    historique = _construire_historique(ligne)
    type_ligne = historique["type_ligne"].mode()[0]
    rotations = _rotations_par_jour(ligne)

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

    chemin = []
    for _ in range(horizon):
        feat = _ajouter_features(historique)
        # IMPORTANT : on exclut 'bus' du dropna. Les jours predits qu'on ajoute a l'historique
        # n'ont pas de 'bus' connu (NaN) ; un dropna() global les effacerait a chaque iteration,
        # et le modele recevrait toujours la meme fenetre -> prediction constante jour apres jour.
        colonnes_essentielles = [c for c in feat.columns if c != "bus"]
        feat = feat.dropna(subset=colonnes_essentielles)
        derniere_fenetre = feat[FEATURES].iloc[-WINDOW_SIZE:]
        X = scaler.transform(derniere_fenetre)[np.newaxis, :, :]
        y_scaled = modele.predict(X, verbose=0).flatten()[0]

        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, 0] = y_scaled
        passagers_predits = float(scaler.inverse_transform(dummy)[0, 0])

        prochaine_date = historique.index[-1] + timedelta(days=1)

        # regle metier : la ligne 73 ne fonctionne pas en juillet/aout, quoi que dise le modele
        if str(ligne) == "73" and prochaine_date.month in (7, 8):
            passagers_predits = 0.0

        bus_recommandes = _bus_necessaires(passagers_predits, prochaine_date, type_ligne, rotations)

        chemin.append({
            "date": prochaine_date.strftime("%Y-%m-%d"),
            "passagers_predits": round(passagers_predits, 1),
            "bus_recommandes": bus_recommandes,
        })

        historique.loc[prochaine_date] = {
            "passagers": passagers_predits, "type_jour": "LV",
            "type_ligne": type_ligne, "bus": np.nan,
        }

    return type_ligne, chemin


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

    # on ne renvoie que les jours reellement demandes par l'utilisateur (>= date_debut)
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


@app.get("/historique/{ligne}")
def consulter_historique(ligne: str, date_debut: str = None, date_fin: str = None):
    """Consulte les donnees REELLES (passees) d'une ligne, avec filtre de dates optionnel.
    Exemples :
    - /historique/12                                   -> tout l'historique disponible
    - /historique/12?date_debut=2025-01-01&date_fin=2025-01-31
    """
    donnees = _df[_df["ligne"] == ligne].sort_values("date")
    if len(donnees) == 0:
        raise HTTPException(404, f"Aucune donnee historique trouvee pour la ligne '{ligne}'")

    if date_debut:
        donnees = donnees[donnees["date"] >= pd.Timestamp(date_debut)]
    if date_fin:
        donnees = donnees[donnees["date"] <= pd.Timestamp(date_fin)]

    resultat = donnees[["date", "passagers", "bus", "type_jour", "type_ligne"]].copy()
    resultat["date"] = resultat["date"].dt.strftime("%Y-%m-%d")

    return {
        "ligne": ligne,
        "nb_jours": len(resultat),
        "historique": resultat.to_dict(orient="records"),
    }
