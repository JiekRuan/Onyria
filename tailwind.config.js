/** @type {import('tailwindcss').Config} */
const defaultTheme = require('tailwindcss/defaultTheme')

module.exports = {
  content: [
    "./DreamProject/**/*.html",
    "./diary/templates/**/*.html",
    "./accounts/templates/**/*.html",
    "./frontend/**/*.js",
    "./js/**/*.js",
  ],
  theme: {
    extend: {
      // 1) On remplace font-sans par Schibsted Grotesk (par défaut partout)
      fontFamily: {
        sans: ['"Schibsted Grotesk"', ...defaultTheme.fontFamily.sans],
        // 2) (optionnel) on garde une utilitaire dédiée au cas où
        grotesk: ['"Schibsted Grotesk"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
