# ── Build stage ────────────────────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --frozen-lockfile

COPY . .

# Optional build-time URL (runtime config.js overrides this on Railway).
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ── Serve stage ────────────────────────────────────────────────────────────────
FROM node:18-alpine

RUN npm install -g serve

COPY --from=builder /app/dist /app/dist

EXPOSE 3000

# Write runtime API URL from Railway env at container start (no rebuild needed).
CMD ["sh", "-c", "node -e \"require('fs').writeFileSync('/app/dist/config.js', 'window.__DP_API__=' + JSON.stringify(process.env.VITE_API_URL || '') + ';')\" && exec serve -s /app/dist -l ${PORT:-3000}"]
