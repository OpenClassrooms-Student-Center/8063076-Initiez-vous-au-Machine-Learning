# Quiz

Le department de la police du la ville de xxx a decidé de predire le caractere criminel des gens basé sur leur visage

Dubitatif, vous êtes en charge de la partie ML du projet

vous fourbissez votre scikit learn et construisez un super modele

Le jeux de donnée:
- physique: caracteristique du visage
- social: etude, salaire, emploi, statut marital
- comportemental: deplacements, achats

1e phase: les donnees du visage (votre telephone vous reconnait, il doit y avoir du signal non ?)


Q1) quel est un bon benchmark pour comparer l'efficacite de votre modele
- coin flip
- regle sur longueur des oreilles
- regle sur homme femme
- 20%

solution: 1 et 3
1: si on bat un modele totallement aleatoire on aura gagné
2: aucune correlation possible entre longeur des oreilles et comportement criminel
3: statistiquement les hommes sont plus a meme d'avoir un comportement criminel donc c'est pas absurde de prendre ca comme point de depart

q2) on a pas de colonne labelisée criminel, quelle strategie employer
- on va recruter des gens pour tagger les images en fonction de leur casier judiciaire
- on va tagger de facon aleatoire
- on va supposer que personne n'est criminel
- on va faire du clustering

solution
1: oui cela prend du temps mais c'est une etape necessaire pour avoir un dataset supervisé
2: ca ne sert a rien
3: meme si c'est la bonne approche legalement et socialement, cela ne nous permettra pas d'avoir un dataset
4: le modele trouvera surement de s groupes de gens par afffinite mais cela ne veut pas dire qu'ils siont plus ou moins criminel

q3) on a une colonne criminel, et les caracteristiques suivantes
- couleur des yeux
- nombre de poils sur les oreille
- presence de taches de rousseurs

on entraine un modele qui predit a 55% correctement le comportement criminel

- c'est un fluke une erreur de modelisation
- incroyable on a montré que la combinaison de ces caracteristiques permet de predire mieux qu'un coin flip
- ces variables sont en fait des proxi pour d'autre variables plus significatives.
- avec la longueur du menton on aurait une disruption dans la force!


q4) depuis 2020, le fisc est autorise a analyser les reseaux sociaux des personnes tres publqiues pour valider netre autre des declarations erroné de residence fiscale. en gros ils veulent s'arrurer que les contribuables domocilie a l'etranger reside bien a letranger et que c'est pas simplemet une question de reduction d'impot

Comment le fisc peut il s'y prendre

https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000043129895

A partir de cet échantillon, sont collectées, à partir des contenus visés à l'article 2, les données suivantes :
a) Les données d'identification des titulaires des pages internet analysées ;
b) Les contenus des pages permettant d'identifier des lieux géographiques qui peuvent notamment être des écrits, des images, des photographies, des sons, des signaux ou des vidéos.

q5) limite de l'approche blackbox

Alphonse demande un credit et se le vois refusé
Il demande a la banque pourquoi ce refus

la banque ne sait pas lui repondre elle a utilsié un modele BB qui ne permet pas de savoir comment chqaue decision a ete prise

- c;est illegale et Alphone peut demander explication
- c'est comme ca et la banque n;as pas a s'expliquer
- il y a des protections contre la discrimination dnas les algo de ML
- tant qu'il n'a rien se reprocher il n;y a pas de probleme (pas de fumee sans feu)

q6) le modele de prediction est en production
mais on s'appercoit que ses perfo decroient fortement depuis 2 j
Que se passe til probablement
- les donnees

C'est bien connu tous les malfrats ont des lunettes noires des cheveux gominé et des garndes moustaches
Notre modele le sait bie et a bien identifie ces patterns
What could go wrong

- au mois de novembre on se laisse pousser la moustache
- en ete on porte des lunettes
- en decembre on porte des pulls de noels
- en automne il pleut et on a des chapeaux

q7) comment choisir son modele

- celui qui est rapide car il doit etre integré dans les lunettes des policiers donc fonctionner avec pe d'energie
- les donnees ne doivent pas
- NN pour le traitement d'image
-

q8) gradient simplifie


q9) on entraine un model black box de detection des feux de forets sur la foret francaise mediterrane

on applique ce modele au nord de la france

what could go wrong
quel scenario est plausible

- les foret ne sont pas pareils et le modele ne va pas bien fonctionner
- le modele repose sur la detection de feux donc un feux est un feux quelque au esoit la foert donc ca marcherca
- il est plus rapide de deployer le modele sur le territoire ...
- impossible de savoir il faut tester d'abord sur un echantillon avec les forets du nord de la france

q10) meme probleme mais avec une regression lineaire qui explicite bien les parametres
sachant que ce que regarde le modele est la presence de fumee et de forte lueur
- les coef seront differents


q11) binaire ou proba

on entraine un modele variable cible : criminel / pas criminel
c'est
- classification binaire?
- regression
- clustering
- classification ordinale

on entraine un modele variable cible: probab de commetre un crime
- regression
- classifcation ordinale


variable cible: un pe, moyen, beaucoup
classification ordonneee
