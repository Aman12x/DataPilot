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
COPY runtime-config.js /app/runtime-config.js

EXPOSE 3000

# Writes dist/config.js (runtime API URL) and dist/serve.json (security headers)
# at container start, so neither needs a rebuild and the CSP's connect-src can
# name the API origin from VITE_API_URL.
CMD ["sh", "-c", "node /app/runtime-config.js && exec serve -s /app/dist -l ${PORT:-3000}"]
