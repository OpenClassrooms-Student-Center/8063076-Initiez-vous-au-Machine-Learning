# Quiz P2

10h du matin, -30 degré et un vent a 140km/h sur l'ile Biscoe dans l'archipel Palmer en Antartique. Vous n'avez pas vu le soleil depuis 3 mois. Bien camouflé.e dans la neige, vous observez à la jumelle un groupe de manchots tout en notant rapidement leurs mesurations sur votre tablette.

Vous faites partie de l'équipe du Pr Kristen Gorman https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php
de la station Palmer https://pallter.marine.rutgers.edu/  et vous adorez les manchots!

Votre travail de recherche porte sur

"l'impact de la variabilité de la glace de mer hivernale sur la recherche de nourriture pré-nuptiale des mâles et des femelles manchots"

De retour à la station, vous chargez le dataset que vous avez compilé avec vos collègues et vous vous mettez au travail.

Note sur la traduction penguins => manchots: en francais, les pingouins vivent dans l'hémisphère nord et ils peuvent voler ! Quant aux manchots, ils ne peuvent pas voler et ils vivent dans l'hémisphère sud.

## Le dataset

insert art

Artwork by @allison_horst

Le dataset a bien été compilé sur le terrain par l'équipe du Dr Gorman. Il est analysé par Allison Horst en détail (avec R) sur https://allisonhorst.github.io/palmerpenguins/

Il est composé de 344 échantillons

des variables categoriques

- species, 3 familles de manchots: Adelie (152), Gentoo (124), Chinstrap (68)
- island, sur 3 iles: Biscoe (168) , Dream (124), Torgersen (52)
- sex: 168 mâles et 165 femelles
- year: collectés entre 2007 et 2009

Des variables numériques

- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g

Pour charger le dataset en python

pip install
