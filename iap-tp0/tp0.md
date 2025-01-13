---
title: TP0 -- Création d'un environnement de travail
author: 
    - Introduction à l'apprentissage profond
    - M2 informatique, Université Paris Cité, 2024-2025
babel-lang: french
---

Voici quelques étapes que vous pouvez suivre pour configurer un environnement de travail pour ce cours.

1. Installez une distribution Anaconda adaptée à votre système d'exploitation; voir [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)

*Remarque :* Après la procédure d'installation, `conda` s'activera automatiquement à chaque fois que vous ouvrirez une nouvelle session terminal. Vous pouvez le remarquer grâce à l'apparition de la chaîne "(base)" devant votre command prompt. 
Je vous recommande de désactiver cette fonctionnalité en exécutant :

`conda config --set auto_activate_base false`  

Voir [cette discussion](https://stackoverflow.com/a/54560785).

2. Créez un environnement nommé `iap` qui utilise Python 3.13 en exécutant :  

`conda create --name iap python=3.13`

3. Activez cet environnement en exécutant :  

`conda activate iap`

Vous devriez maintenant voir une mention supplémentaire "(iap)" devant votre invite de commande, comme ceci: 

`(iap) sam@samdell:~$`

Vérifiez que la commande `python --version` affiche bien une version Python 3.13.x.

4. Utilisez `pip` pour installer les packages nécessaires dans cet environnement en exécutant :  

`pip install pandas torch numpy jupyter plotly matplotlib`

5. Si vous devez désactiver l'environnement, fermez simplement la session terminal ou exécutez:  

`conda deactivate iap`
