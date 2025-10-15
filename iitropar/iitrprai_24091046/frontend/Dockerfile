FROM node:20

WORKDIR /app

COPY package*.json ./
RUN npm install

# Copy app
COPY . ./

EXPOSE 3000

CMD ["npm", "start"]
