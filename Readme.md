### Compilation du CSS Tailwind

Pour compiler le CSS Tailwind localement :

**Compilation unique** (avant déploiement) :
```bash
npm run build:css
```

**Compilation automatique** (mode développement) :
```bash
npm run watch:css
```

**Important** : Le fichier généré `static/css/tailwind.min.css` est ignoré par Git et ne doit pas être committé. Il est automatiquement généré lors du déploiement sur Render via la Build Command.

**Déploiement** : En pré-production et production, le CSS est compilé automatiquement via la Build Command de Render.