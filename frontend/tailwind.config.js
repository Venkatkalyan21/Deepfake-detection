/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["Fira Code", "JetBrains Mono", "monospace"],
      },
      colors: {
        primary: {
          400: "#00f0ff", // Electric Cyan
          500: "#00c3ff",
          600: "#0096ff",
        },
        danger: {
          400: "#ff2a2a",
          500: "#e60000",
          600: "#cc0000",
        },
        success: {
          400: "#00ff88",
          500: "#00cc6a",
          600: "#009950",
        },
        dark: {
          800: "#111827",
          900: "#0a1128", // Navy/black base
          950: "#040814",
        },
      },
      animation: {
        "scanline": "scanline 8s linear infinite",
        "fade-in": "fadeIn 0.3s ease-out",
        "glitch": "glitch 1s linear infinite",
      },
      keyframes: {
        fadeIn: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        scanline: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(1000%)" },
        },
      },
      backgroundImage: {
        "grid-pattern": "linear-gradient(to right, #ffffff05 1px, transparent 1px), linear-gradient(to bottom, #ffffff05 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};
