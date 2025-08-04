/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./DreamProject/**/*.html",        // Tous tes templates Django dans le projet principal
    "./diary/templates/**/*.html",     // Templates dans l'app diary
    "./accounts/templates/**/*.html",  // Templates dans l'app accounts
  ],
  theme: {
    extend: {
      fontFamily: {
        grotesk: ['"Schibsted Grotesk"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
