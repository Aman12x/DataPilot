# ── Build stage ────────────────────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --frozen-lockfile

COPY . .

# VITE_API_URL is injected at build time via Railway env var
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ── Serve stage ────────────────────────────────────────────────────────────────
FROM node:18-alpine

RUN npm install -g serve

COPY --from=builder /app/dist /app/dist

EXPOSE 3000

CMD ["sh", "-c", "serve -s /app/dist -l ${PORT:-3000}"]
