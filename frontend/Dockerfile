# ── build ────────────────────────────────────────────────────────
FROM node:22-alpine AS build

WORKDIR /app

ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Dependencies first, so source edits do not invalidate the install layer.
COPY package.json package-lock.json* ./
# `npm ci` installs exactly the lockfile: reproducible, and it fails loudly
# if package.json and the lockfile have drifted apart.
RUN npm ci --no-audit --no-fund

COPY . .
RUN npm run build

# ── serve ────────────────────────────────────────────────────────
FROM nginxinc/nginx-unprivileged:1.27-alpine

# The unprivileged image already runs as uid 101 and listens on 8080, so no
# root-owned pid file, cache directory or privileged port is needed.
USER nginx

COPY --chown=nginx:nginx security-headers.conf /etc/nginx/conf.d/security-headers.conf
COPY --chown=nginx:nginx nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build --chown=nginx:nginx /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/nginx-health || exit 1

STOPSIGNAL SIGQUIT
