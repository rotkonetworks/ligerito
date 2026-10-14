FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html style.css site.webmanifest /usr/share/nginx/html/
COPY favicon.ico favicon-16x16.png favicon-32x32.png /usr/share/nginx/html/
COPY apple-touch-icon.png android-chrome-192x192.png android-chrome-512x512.png /usr/share/nginx/html/
COPY a/ /usr/share/nginx/html/a/
EXPOSE 8080
