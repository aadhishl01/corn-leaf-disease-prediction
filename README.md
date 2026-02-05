# Black and White E-Commerce Application

A full-stack e-commerce application built with the MERN stack (MongoDB, Express, React, Node.js).

## Project Overview

"Black and White" is a modern e-commerce platform featuring a responsive frontend and a robust backend.
It includes features for user authentication, product browsing, cart management, and order processing.

### Key Features

-   **Frontend**: Built with React (Vite) and Tailwind CSS for a fast and styling user interface.
-   **Backend**: Node.js and Express server handling API requests.
-   **Database**: MongoDB for storing user and product data.
-   **Image Storage**: Cloudinary integration for product image management.
-   **Authentication**: JWT-based authentication for secure user login/signup.
-   **Deployment**: Ready for deployment with separate frontend and backend configurations.

## key Technologies

### Frontend
-   **React** (v18+)
-   **Vite** (Build tool)
-   **Tailwind CSS** (Styling)
-   **React Router DOM** (Navigation)
-   **Context API** (State Management)
-   **Firebase** (Integration)

### Backend
-   **Node.js** & **Express.js**
-   **MongoDB** (Database)
-   **Mongoose** (ODM)
-   **Cloudinary** (Image Uploads)
-   **JWT** (Authentication)
-   **Bcrypt** (Password Hashing)

## Project Structure

```
d:/ecommerce app/
├── frontend/           # React frontend application
│   ├── src/            # Source code (components, pages, context)
│   ├── public/         # Static assets
│   └── package.json    # Frontend dependencies
│
├── backend/            # Express backend application
│   ├── config/         # Configuration files (DB, Cloudinary)
│   ├── models/         # Mongoose models
│   ├── routes/         # API routes
│   ├── server.js       # Backend entry point
│   └── package.json    # Backend dependencies
│
└── README.md           # Project documentation
```

## Setup Instructions

### Prerequisites
-   Node.js installed
-   MongoDB installed or a MongoDB Atlas URI
-   Cloudinary Account

### 1. Backend Setup

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Create a `.env` file in the `backend` directory and add your configuration:
    ```env
    MONGODB_URI=your_mongodb_connection_string
    CLOUDINARY_CLOUD_NAME=your_cloud_name
    CLOUDINARY_API_KEY=your_api_key
    CLOUDINARY_API_SECRET=your_api_secret
    JWT_SECRET=your_jwt_secret
    ```
4.  Start the server:
    ```bash
    npm start
    ```
    The server will run on `http://localhost:4000`.

### 2. Frontend Setup

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    The application will be accessible at `http://localhost:5173` (or the port shown in your terminal).

## API Endpoints

-   **User Routes**: `/api/user` (Login, Register)
-   **Product Routes**: `/api/product` (List, Add, Delete products)

## License

This project is licensed under the ISC License.
