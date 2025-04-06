/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  safelist: ['font-glitch'],
  theme: {
    extend: {
      keyframes: {
        modalDrop: {
          "0%": { transform: "scale(0.9)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
      },
      animation: {
        modalDrop: "modalDrop 0.2s ease-out",
      },
      fontFamily: {
        'glitch': ['LLDEtechnoGlitch', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
