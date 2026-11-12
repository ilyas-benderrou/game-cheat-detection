# Explication simple du projet « Détection de triche » (pour préparer l'entretien)

## 1. L'idée en une phrase

On regarde comment joue une personne pendant une partie (précision, headshots, ratio
kills/morts…) et on essaie de deviner automatiquement si elle triche.

## 2. Les données

- Un fichier CSV de **5 000 lignes**. **Une ligne = une session de jeu** d'un joueur.
- **12 colonnes**, par exemple :
  - `Accuracy` : la précision. 0,35 = 35 % des balles touchent la cible.
  - `Headshot_Rate` : part des tirs dans la tête. 0,25 = 1 tir sur 4.
  - `KD_Ratio` : kills divisés par morts. 2,0 = le joueur tue 2 fois plus qu'il ne meurt.
  - `APM` : *actions par minute* (clics + touches). 150 = 150 actions par minute.
  - `Cheat_Likelihood` : **la cible**, ce qu'on veut prédire. 0 = normal, 1 = suspect.
- **Déséquilibre** : 4 771 normaux pour seulement 229 suspects (4,6 %).
  Exemple : si un modèle dit toujours « normal », il a 95,4 % de bonnes réponses… mais il ne
  détecte aucun tricheur. C'est pour ça qu'on ne regarde pas seulement l'*accuracy*.

## 3. Le nettoyage

La colonne `Items` (objet acheté) est vide pour 1 049 sessions. En réalité le fichier
contient le mot `None`, que pandas (la bibliothèque qui lit le tableau) comprend comme
« valeur manquante ». On ne supprime pas ces lignes : on écrit `No_Item` à la place.
Exemple : une ligne `Items = (vide)` devient `Items = No_Item`.

## 4. Le feature engineering (créer de nouvelles variables)

Une **feature** (variable) est une colonne que le modèle utilise. On en fabrique trois :

| Nouvelle variable | Formule | Exemple |
|---|---|---|
| `Aim_Score` | Accuracy × Headshot_Rate | 0,9 × 0,9 = 0,81 (très suspect) ; 0,35 × 0,25 ≈ 0,09 (normal) |
| `Combat_Score` | KD_Ratio × Accuracy | 12 × 0,9 = 10,8 (suspect) ; 1,7 × 0,35 ≈ 0,6 (normal) |
| `Activity_Score` | APM / durée de session | 150 / 30 min = 5 |

## 5. Le prétraitement

- **On enlève les identifiants** (`Player_ID`, `Game_Session_ID`) : ils ne disent rien du
  comportement.
- **Standardisation** (`StandardScaler`) : on met chaque variable numérique sur la même
  échelle (moyenne 0, écart-type 1). Sinon l'APM (≈150) écraserait l'Accuracy (≈0,35).
- **One-hot encoding** : on transforme un texte en colonnes de 0/1.
  Exemple : `Region = EU-West` devient `Region_EU-West = 1`, `Region_NA-West = 0`, …
- **Split 80/20 stratifié** : 80 % des lignes pour apprendre (*train*, 4 000), 20 % pour
  tester (*test*, 1 000). « Stratifié » = on garde la même proportion de tricheurs (4,6 %)
  dans les deux parties. Le test contient 954 normaux et 46 suspects.

## 6. Les 5 modèles

### Modèles **supervisés** (ils apprennent avec les réponses)
On leur montre des exemples avec l'étiquette « tricheur / normal ».

1. **Régression logistique** : calcule une somme pondérée des variables puis la transforme
   en probabilité entre 0 et 1. Exemple : probabilité 0,97 → on prédit « tricheur ».
   `class_weight="balanced"` = une erreur sur un tricheur (rare) compte plus qu'une erreur
   sur un joueur normal.
2. **Random Forest (forêt aléatoire)** : 300 arbres de décision qui votent. Un arbre pose
   des questions du type « Accuracy > 0,6 ? ». La majorité des votes donne la réponse.

### Modèles **non supervisés / détection d'anomalies** (ils n'ont jamais vu de tricheur)
On les entraîne **uniquement sur les joueurs normaux**. Ils apprennent « à quoi ressemble
un joueur normal » et signalent ce qui est trop différent. C'est utile dans la vraie vie,
car on a rarement des étiquettes de tricheurs.

3. **Isolation Forest** : on coupe les données au hasard ; un point bizarre est isolé en
   très peu de coupes. Exemple : un joueur avec 99 % de headshots est isolé tout de suite.
4. **One-Class SVM** : dessine une frontière autour des joueurs normaux ; ce qui est dehors
   est suspect.
5. **Autoencodeur (MLP)** : un petit réseau de neurones qui doit **recopier** son entrée en
   passant par un goulot de 8 neurones (16 → 8 → 16). Il apprend à bien recopier les
   joueurs normaux. Pour un tricheur, la copie est mauvaise → **erreur de reconstruction**
   grande → alerte.

**Le seuil** : pour ces 3 modèles il faut décider à partir de quand on lance une alerte.
On a choisi de signaler les 4,6 % de joueurs normaux les plus « bizarres » (le taux de
triche connu). Conséquence : environ 43 à 50 joueurs honnêtes sont signalés dans le test.

## 7. Les métriques (comment on note un modèle)

Matrice de confusion = tableau 2×2 :

| | prédit normal | prédit suspect |
|---|---|---|
| **vrai normal** | TN (vrai négatif) | FP (faux positif = fausse alerte) |
| **vrai suspect** | FN (faux négatif = tricheur raté) | TP (vrai positif) |

- **Precision** = TP / (TP + FP) : parmi les joueurs qu'on accuse, combien trichent vraiment ?
  Exemple Isolation Forest : 46 / (46 + 43) = 0,517 → une alerte sur deux est juste.
- **Recall (rappel)** = TP / (TP + FN) : parmi les vrais tricheurs, combien sont attrapés ?
  Exemple : 46 / 46 = 1,0 → aucun tricheur raté.
- **F1** : moyenne « harmonique » de precision et recall ; elle est basse si l'une des deux
  est basse.
- **ROC-AUC** : probabilité qu'un tricheur pris au hasard ait un score plus suspect qu'un
  joueur normal pris au hasard. 0,5 = hasard, 1,0 = classement parfait. Elle ne dépend
  pas du seuil.

## 8. Les résultats (vrais chiffres obtenus en lançant le code)

| Modèle | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Régression logistique | 1,000 | 1,000 | 1,000 | 1,000 |
| Random Forest | 1,000 | 1,000 | 1,000 | 1,000 |
| Isolation Forest | 0,517 | 1,000 | 0,682 | 0,995 |
| One-Class SVM | 0,479 | 1,000 | 0,648 | 1,000 |
| Autoencodeur | 0,517 | 1,000 | 0,682 | 1,000 |

- **Validation croisée à 5 plis** (*5-fold cross-validation*) : on coupe les données en 5
  morceaux, on entraîne 5 fois en testant à chaque fois sur un morceau différent. Résultat :
  1,000 partout pour les deux modèles supervisés → ce n'est pas un coup de chance du split.
- **Importance des variables** (Random Forest) : Combat_Score, KD_Ratio, Aim_Score, Accuracy
  et Headshot_Rate font ~98 % des décisions. L'APM compte très peu.

## 9. Pourquoi 100 % est « trop beau » (point clé à dire en entretien)

Le script vérifie les intervalles de valeurs :

- Accuracy des normaux : 0,20 à 0,50 ; des suspects : 0,80 à 0,99.
- Headshot_Rate : 0,10–0,40 contre 0,80–1,00.
- KD_Ratio : 0,5–3,0 contre 5,1–19,9.

Les deux groupes **ne se chevauchent jamais**. Une seule règle « Accuracy > 0,65 → tricheur »
suffit à avoir 100 %. Le dataset est très probablement **synthétique** (généré par
ordinateur), donc trop facile.

**Expérience d'ablation** (on enlève volontairement des variables pour voir ce qui reste) :
sans les 5 variables de visée, la Random Forest tombe à F1 = 0,283 et ROC-AUC = 0,745, la
régression logistique à F1 = 0,130 et ROC-AUC = 0,698. Donc sans statistiques
« absurdes », trouver un tricheur est difficile.

Dans un vrai jeu, c'est plus dur car : les tricheurs discrets ressemblent à de très bons
joueurs, les étiquettes (bannissements) sont rares et en retard, les triches changent, et il
faudrait des données fines (mouvements de souris, temps de réaction tir par tir).

## 10. Phrase de conclusion pour l'entretien

« Sur ce dataset, les modèles supervisés atteignent 100 %, mais j'ai montré que c'est parce
que les classes sont parfaitement séparables. Les détecteurs d'anomalies, entraînés sans
aucun tricheur, attrapent tous les tricheurs avec environ une fausse alerte sur deux, ce qui
est plus proche d'un usage réel. Pour une thèse, la suite serait de travailler sur des
données plus réalistes, séquentielles, et sur le choix du seuil d'alerte. »

## Petit lexique

- **Pipeline** : chaîne d'étapes automatiques (prétraitement + modèle) appliquée pareil au
  train et au test.
- **Overfitting (surapprentissage)** : le modèle apprend par cœur le train et échoue sur de
  nouvelles données. Ici ce n'est pas le problème : c'est le dataset qui est trop simple.
- **joblib** : format pour sauvegarder un modèle entraîné sur le disque (`.joblib`).
- **random_state = 42** : graine du hasard fixée, pour obtenir exactement les mêmes résultats
  à chaque exécution.
