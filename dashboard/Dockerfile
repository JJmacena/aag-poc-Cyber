# Dashboard - React Frontend
FROM node:20-alpine AS builder

WORKDIR /app

COPY dashboard/frontend/package*.json ./
RUN npm install

COPY dashboard/frontend/ ./
RUN ls -la /app/src/

RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=builder /app/build /usr/share/nginx/html
COPY infra/nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]