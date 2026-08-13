"""
GENERATION DES PREVISIONS - VERSION FINALE
===========================================
Projet PFA ALSA Agadir - Outil d'aide a la decision (prevision demande voyageurs + bus).

CE SCRIPT FAIT UNE SEULE CHOSE : generer les previsions (passagers + bus recommandes)
pour toutes les lignes actives, sur un horizon donne, avec les choix definitifs retenus
apres comparaison (voir justifications ci-dessous). Il n'entraine RIEN : il charge les
modeles deja entraines et sauvegardes par le notebook PFA_ALSA_LSTM.ipynb (section 8).


================================================================================
 JUSTIFICATION DES CHOIX (a reprendre dans le rapport / la soutenance)
================================================================================

1) MODELE LSTM PAR LIGNE (section 8 du notebook), PAS le modele global (section 9)
   - Chaque ligne a des dynamiques propres (frequentation, saisonnalite, jours de
     forte/faible affluence) qu'un modele dedie apprend mieux qu'un modele unique
     partage entre 38 lignes.
   - La comparaison faite dans le notebook (`comparaison_approches`, section 9) ne
     montre pas d'avantage systematique du modele global ; il n'est donc pas retenu
     comme methode principale, seulement comme solution de repli pour une ligne sans
     historique suffisant.

2) ESTIMATION DES BUS PAR CALCUL DIRECT (Methode 1), PAS un LSTM entraine sur 'bus'
   - Methode retenue : passagers predits (LSTM) -> bus_necessaires() = formule
     capacite x rotations.
   - Plus interpretable et defendable a l'oral (chaque etape s'explique), plus robuste
     dans le temps (si ALSA change sa politique de bus, il suffit de changer
     CAPACITE_PAR_ERE, pas besoin de reentrainer), et ne reproduit pas aveuglement
     d'eventuelles inefficacites deja presentes dans l'historique de deploiement.
   - La Methode 2 (LSTM entraine directement sur 'bus', section 7.1 du notebook) reste
     un point de comparaison academique interessant pour le rapport, mais n'est pas
     utilisee en production ici.

3) ROTATIONS "AFFINEES" (section 7.2) : duree de trajet reelle quand connue,
   repli automatique sur la calibration empirique (passagers/bus) sinon.

4) FENETRE D'ENTREE WINDOW_SIZE = 30 jours (contrainte fixee : plus de contexte
   temporel pour le LSTM que les 14 jours initiaux).

5) JOURS FERIES CIVILS (dates fixes, Moharram, Mawlid) traites comme des dimanches (meme
   encodage du jour de semaine). L'AID (Fitr/Adha) est traite A PART, PAS assimile a un
   dimanche : son comportement est bien plus marque (frequentation souvent < 50% de la
   normale) et est corrige explicitement via calculer_facteur_aid(), a partir des vrais
   jours d'Aid 2024/2025/2026 de la ligne concernee. declarer_jour_aid() permet de marquer
   manuellement une date comme Aid (ex: annee future non encore dans le calendrier).

6) LIGNE 73 : exclue en juillet/aout (a l'arret l'ete, regle metier connue) - force
   a 0 passager/0 bus si jamais un appel tombe sur ces mois.

7) BUS MINIMUM = 1 (jamais 0.5 seul) des qu'il y a du trafic : une ligne a faible
   frequentation tourne quand meme toute la journee (matin ET soir).

8) PASSAGERS PREDITS : toujours un entier positif ou nul (jamais negatif, jamais a virgule -
   un nombre de personnes ne peut pas etre fractionnaire ni negatif).


================================================================================
 ARBORESCENCE ATTENDUE (a cote de ce script, copiee depuis Drive)
================================================================================
    data/
        BD_EXPLOITATION_clean.csv
    models/
        lstm_ligne_<NOM>.keras
        scaler_ligne_<NOM>.pkl
    reports/
        (previsions_finales_<date>.csv / .xlsx generes ici)

Usage :
    python generer_previsions_finales.py --horizon 90
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
DATA_PATH = "data/BD_EXPLOITATION_clean.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

WINDOW_SIZE = 30
FEATURES = ['passagers', 'is_vacances', 'is_weekend', 'is_ferie', 'is_eid',
            'jour_0', 'jour_1', 'jour_2', 'jour_3', 'jour_4', 'jour_5', 'jour_6',
            'sin_mois', 'cos_mois',
            'lag_1', 'lag_7', 'lag_365', 'lag_730',
            'rolling_mean_7', 'rolling_mean_30']

# --------------------------------------------------------------------------
# Calendrier des jours feries marocains (identique au notebook, section 3.1)
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
dates_feries_fixes = [pd.Timestamp(annee, mois, jour)
                       for annee in ANNEES_COUVERTES for (mois, jour) in JOURS_FERIES_FIXES]
SET_DATES_AID = set(pd.to_datetime(DATES_AID))
SET_DATES_FERIES = set(pd.to_datetime(DATES_AUTRES_FERIES_RELIGIEUX)) | set(dates_feries_fixes)

# Declaration manuelle : permet de marquer un jour comme Aid meme s'il n'est pas (encore) dans
# le calendrier code en dur ci-dessus (utile pour une annee future, ex. 2027, ou pour corriger
# une date). Ex : declarer_jour_aid('2027-03-10')
JOURS_AID_DECLARES = set()


def declarer_jour_aid(date):
    JOURS_AID_DECLARES.add(pd.Timestamp(date).normalize())


def annuler_declaration_aid(date):
    JOURS_AID_DECLARES.discard(pd.Timestamp(date).normalize())


def est_jour_ferie_civil(date):
    """Jours feries CIVILS uniquement (dates fixes + Moharram/Mawlid) : leur comportement
    est assimile a un dimanche. L'Aid n'en fait PAS partie : voir est_jour_aid ci-dessous -
    son comportement est bien plus marque qu'un simple dimanche (cf calculer_facteur_aid)."""
    return date.normalize() in SET_DATES_FERIES


def est_jour_aid(date):
    """Aid El Fitr / Aid El Adha : comportement PROPRE, pas assimile a un dimanche. Garde son
    vrai jour de semaine dans les features ; corrige a part via calculer_facteur_aid()."""
    d = date.normalize()
    return (d in SET_DATES_AID) or (d in JOURS_AID_DECLARES)


def est_jour_ferie(date):
    """Jour ferie au sens large (civil OU Aid) - feature generale 'jour special'."""
    return est_jour_ferie_civil(date) or est_jour_aid(date)


# --------------------------------------------------------------------------
# Capacite des bus et rotations (section 7 / 7.2 du notebook)
# --------------------------------------------------------------------------
CAPACITE_PAR_ERE = {
    "avant_2026": {"URB": 100, "REG": 90},
    "2026_et_plus": {"URB": 90, "REG": 75},
}

# Duree d'un trajet ALLER SIMPLE, en minutes, fournie par ALSA
DUREE_TRAJET_MIN = {
    '1': 25, '2': 40, '3': 50, '5': 60, '6': 40, '8': 46, '9': 30, '10': 30, '11': 45,
    '12': 65, '13': 30, '14': 30, '15': 50, '16': 60, '20': 45, '21': 40, '22': 60,
    '23': 40, '24': 10, '26': 65, '31': 60, '32': 60, '33': 90, '35': 40, '36': 70,
    '37': 60, '38': 60, '39': 30, '40': 70, '41': 80, '42': 100, '43': 60, '95': 50,
    '97': 45, '98': 40, '73': 45, '27': 30, 'AE': 45, '99': 30,
}


def capacite_du_jour(date, type_ligne):
    era = "avant_2026" if date.year < 2026 else "2026_et_plus"
    return CAPACITE_PAR_ERE[era][type_ligne]


def arrondir_au_demi(x):
    return np.round(x * 2) / 2


def calibrer_rotations_par_jour(historique_df, type_ligne):
    """Repli empirique : mediane de passagers/(bus x capacite) sur l'historique."""
    h = historique_df[historique_df['bus'] > 0].copy()
    if len(h) < 10:
        return 1.0
    cap_nominale = np.array([capacite_du_jour(d, type_ligne) for d in h.index])
    ratio = h['passagers'] / (h['bus'] * cap_nominale)
    return float(ratio.median())


def calibrer_rotations_theorique(historique_df, duree_aller_min):
    """Methode retenue en priorite : heures de service reelles par bus (horas/bus)
    divisees par la duree d'un aller-retour (2 x duree_aller_min)."""
    h = historique_df[historique_df['bus'] > 0]
    if len(h) < 10 or duree_aller_min is None:
        return None
    heures_par_bus = (h['horas'] / h['bus']).median()
    if not np.isfinite(heures_par_bus) or heures_par_bus <= 0:
        return None
    rotations = heures_par_bus / ((2 * duree_aller_min) / 60)
    return rotations if np.isfinite(rotations) and rotations > 0 else None


def rotations_finales(ligne, historique_df, type_ligne):
    """Methode affinee : duree reelle si connue et fiable, sinon repli empirique."""
    duree = DUREE_TRAJET_MIN.get(str(ligne))
    rot_theo = calibrer_rotations_theorique(historique_df, duree)
    return rot_theo if rot_theo is not None else calibrer_rotations_par_jour(historique_df, type_ligne)


def bus_necessaires(nb_passagers, date, type_ligne, rotations_par_jour):
    """Methode 1 (calcul direct) - retenue en production. Voir justification en tete de fichier."""
    if nb_passagers <= 0:
        return 0.0
    if not rotations_par_jour or rotations_par_jour <= 0:
        rotations_par_jour = 1.0
    cap = capacite_du_jour(date, type_ligne) * rotations_par_jour
    bus_arrondi = arrondir_au_demi(nb_passagers / cap)
    return max(bus_arrondi, 1.0)   # minimum 1 bus des qu'il y a du trafic (jamais 0.5 seul)


# --------------------------------------------------------------------------
# Preparation des donnees (identique au notebook, section 1/8)
# --------------------------------------------------------------------------
def preparer_serie(df, ligne):
    colonnes_utiles = ['passagers', 'type_jour', 'type_ligne', 'bus', 'horas']
    s = df[df['ligne'] == ligne].sort_values('date').set_index('date')[colonnes_utiles]
    s = s.asfreq('D')
    s['passagers'] = s['passagers'].interpolate(method='time')
    s['type_ligne'] = s['type_ligne'].ffill().bfill()
    s['type_jour'] = s['type_jour'].fillna('LV')
    s['bus'] = s['bus'].interpolate(method='time')
    s['horas'] = s['horas'].interpolate(method='time')
    if str(ligne) == '73':
        s = s[~s.index.month.isin([7, 8])]   # regle metier : arret estival connu
    return s


def add_features(s):
    d = s.copy()
    d['jour_semaine'] = d.index.dayofweek
    d['mois'] = d.index.month

    # jours feries CIVILS (dates fixes + Moharram/Mawlid) : comportement assimile a un dimanche
    d['is_ferie'] = [int(est_jour_ferie_civil(dt)) for dt in d.index]
    # Aid : comportement PROPRE, PAS assimile a un dimanche (voir plus haut)
    d['is_eid'] = [int(est_jour_aid(dt)) for dt in d.index]
    d['is_vacances'] = (d['type_jour'] == 'OTROS').astype(int)

    # jour ferie CIVIL -> encodage force sur "dimanche" ; l'Aid (is_eid) garde son vrai jour
    jour_semaine_ajuste = d['jour_semaine'].where(d['is_ferie'] == 0, 6)
    d['is_weekend'] = np.where(d['is_ferie'] == 1, 1, (d['jour_semaine'] >= 5).astype(int))

    dummies_jour = pd.get_dummies(jour_semaine_ajuste, prefix='jour').astype(int)
    for i in range(7):
        col = f'jour_{i}'
        if col not in dummies_jour:
            dummies_jour[col] = 0
    d = pd.concat([d, dummies_jour], axis=1)

    d['sin_mois'] = np.sin(2 * np.pi * d['mois'] / 12)
    d['cos_mois'] = np.cos(2 * np.pi * d['mois'] / 12)

    d['lag_1'] = d['passagers'].shift(1)
    d['lag_7'] = d['passagers'].shift(7)
    d['rolling_mean_7'] = d['passagers'].shift(1).rolling(7).mean()
    d['rolling_mean_30'] = d['passagers'].shift(1).rolling(30).mean()
    d['lag_365'] = d['passagers'].shift(365).fillna(d['rolling_mean_30'])
    d['lag_730'] = d['passagers'].shift(730).fillna(d['lag_365'])

    # IMPORTANT : on exclut 'bus'/'horas' du dropna (voir justification - bug corrige :
    # sans ca, chaque jour predit lors de la prevision recursive serait efface avant de
    # servir a construire la fenetre suivante -> prediction constante jour apres jour).
    colonnes_essentielles = [c for c in d.columns if c not in ('bus', 'horas')]
    return d.dropna(subset=colonnes_essentielles)


def calculer_facteur_aid(feat_df):
    """Calcule, a partir des vrais jours d'Aid passes de cette ligne, le facteur multiplicatif
    median applique a la frequentation normale (rolling_mean_30). Renvoie None si la ligne
    n'a aucun jour d'Aid observe dans son historique (ligne trop recente)."""
    jours_aid = feat_df[feat_df['is_eid'] == 1]
    if len(jours_aid) == 0:
        return None
    ratio = (jours_aid['passagers'] / jours_aid['rolling_mean_30'].replace(0, np.nan)).dropna()
    return float(ratio.median()) if len(ratio) else None


def inverse_target(scaled_values, scaler, n_features, target_idx=0):
    dummy = np.zeros((len(scaled_values), n_features))
    dummy[:, target_idx] = scaled_values
    return scaler.inverse_transform(dummy)[:, target_idx]


def nom_fichier_sur(ligne):
    return str(ligne).replace(' ', '_').replace('/', '_')


# --------------------------------------------------------------------------
# Prevision recursive pour UNE ligne, jusqu'a un horizon donne (en jours)
# --------------------------------------------------------------------------
def previsions_ligne(df, ligne, horizon):
    safe_name = nom_fichier_sur(ligne)
    chemin_modele = f"{MODELS_DIR}/lstm_ligne_{safe_name}.keras"
    chemin_scaler = f"{MODELS_DIR}/scaler_ligne_{safe_name}.pkl"
    if not (os.path.exists(chemin_modele) and os.path.exists(chemin_scaler)):
        raise FileNotFoundError(f"Pas de modele entraine pour la ligne '{ligne}'")

    modele = load_model(chemin_modele)
    scaler = joblib.load(chemin_scaler)

    historique = preparer_serie(df, ligne)
    type_ligne = historique['type_ligne'].mode()[0]
    rotations = rotations_finales(ligne, historique, type_ligne)
    facteur_aid = calculer_facteur_aid(add_features(historique))

    resultats = []
    for _ in range(horizon):
        feat = add_features(historique)
        derniere_fenetre = feat[FEATURES].iloc[-WINDOW_SIZE:]
        X = scaler.transform(derniere_fenetre)[np.newaxis, :, :]
        y_scaled = modele.predict(X, verbose=0).flatten()[0]
        passagers_predits = float(inverse_target(np.array([y_scaled]), scaler, len(FEATURES))[0])

        prochaine_date = historique.index[-1] + pd.Timedelta(days=1)

        # regle metier : ligne 73 a l'arret en juillet/aout, quoi que dise le modele
        if str(ligne) == '73' and prochaine_date.month in (7, 8):
            passagers_predits = 0.0
        elif est_jour_aid(prochaine_date) and facteur_aid is not None:
            # recalage explicite sur la baisse d'Aid historiquement observee pour cette ligne
            # (plus fiable que ce que le LSTM apprend seul sur les 6 jours d'Aid de l'historique)
            baseline_normale = feat['rolling_mean_30'].iloc[-1]
            passagers_predits = baseline_normale * facteur_aid

        # un nombre de passagers est un entier positif : jamais negatif, jamais a virgule
        passagers_predits = max(0, round(passagers_predits))

        bus_recommandes = bus_necessaires(passagers_predits, prochaine_date, type_ligne, rotations)

        resultats.append({
            'date': prochaine_date,
            'ligne': ligne,
            'passagers_predits': passagers_predits,
            'bus_recommandes': bus_recommandes,
        })

        historique.loc[prochaine_date] = {
            'passagers': passagers_predits, 'type_jour': 'LV',
            'type_ligne': type_ligne, 'bus': np.nan, 'horas': np.nan,
        }

    return pd.DataFrame(resultats)


# --------------------------------------------------------------------------
# Generation pour toutes les lignes actives
# --------------------------------------------------------------------------
def generer_toutes_les_previsions(horizon):
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df['ligne'] = df['ligne'].astype(str)

    toutes_les_lignes = sorted(df['ligne'].unique())
    resultats, echecs = [], []

    for ligne in toutes_les_lignes:
        try:
            prev = previsions_ligne(df, ligne, horizon)
            resultats.append(prev)
            print(f"Ligne {ligne} : OK ({len(prev)} jours generes)")
        except FileNotFoundError as e:
            echecs.append((ligne, str(e)))
            print(f"Ligne {ligne} : ignoree -> {e}")
        except Exception as e:
            echecs.append((ligne, str(e)))
            print(f"Ligne {ligne} : ECHEC -> {e}")

    if not resultats:
        raise RuntimeError("Aucune prevision generee - verifiez le dossier models/.")

    previsions_df = pd.concat(resultats, ignore_index=True)
    previsions_df['date_generation'] = pd.Timestamp.today().strftime('%Y-%m-%d')
    previsions_df = previsions_df[['date', 'ligne', 'passagers_predits', 'bus_recommandes', 'date_generation']]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    horodatage = pd.Timestamp.today().strftime('%Y%m%d')
    chemin_csv = f"{REPORTS_DIR}/previsions_finales_{horodatage}.csv"
    chemin_xlsx = f"{REPORTS_DIR}/previsions_finales_{horodatage}.xlsx"
    previsions_df.to_csv(chemin_csv, index=False)
    previsions_df.to_excel(chemin_xlsx, index=False)

    print(f"\n{len(resultats)} lignes traitees, {len(echecs)} echecs.")
    print(f"Sauvegarde : {chemin_csv}")
    print(f"Sauvegarde : {chemin_xlsx}")
    return previsions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genere les previsions finales (passagers + bus).")
    parser.add_argument("--horizon", type=int, default=90,
                         help="Nombre de jours a prevoir a partir de la derniere donnee reelle (defaut: 90, ~3 mois).")
    args = parser.parse_args()
    generer_toutes_les_previsions(args.horizon)
